use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{Quat, Vec3};

/// Gaussians loaded from a 3DGS-format PLY, converted to linear units.
pub struct SplatCloud {
    pub positions: Vec<Vec3>,
    /// Per-axis standard deviations (linear, not log).
    pub scales: Vec<Vec3>,
    pub rotations: Vec<Quat>,
    /// Opacity in [0, 1] (sigmoid already applied).
    pub opacities: Vec<f32>,
    /// Base color in [0, 1] from the SH DC term.
    pub colors: Vec<[f32; 3]>,
}

impl SplatCloud {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// SH degree-0 basis constant: rgb = 0.5 + C0 * f_dc.
const SH_C0: f32 = 0.282_094_79;

#[derive(Clone, Copy, PartialEq)]
enum PropType {
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
}

impl PropType {
    fn parse(token: &str) -> Option<Self> {
        match token {
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 => 8,
        }
    }

    /// Read this property from `bytes` at `offset` as f32 (little endian).
    fn read_f32(self, bytes: &[u8], offset: usize) -> f32 {
        let b = &bytes[offset..offset + self.size()];
        match self {
            Self::F32 => f32::from_le_bytes(b.try_into().expect("size checked")),
            Self::F64 => f64::from_le_bytes(b.try_into().expect("size checked")) as f32,
            Self::I8 => b[0] as i8 as f32,
            Self::U8 => b[0] as f32,
            Self::I16 => i16::from_le_bytes(b.try_into().expect("size checked")) as f32,
            Self::U16 => u16::from_le_bytes(b.try_into().expect("size checked")) as f32,
            Self::I32 => i32::from_le_bytes(b.try_into().expect("size checked")) as f32,
            Self::U32 => u32::from_le_bytes(b.try_into().expect("size checked")) as f32,
        }
    }
}

struct VertexLayout {
    /// (byte offset, type) per property, in declaration order.
    props: Vec<(String, usize, PropType)>,
    stride: usize,
    count: usize,
}

impl VertexLayout {
    fn offset_of(&self, name: &str) -> Option<(usize, PropType)> {
        self.props
            .iter()
            .find(|(n, ..)| n == name)
            .map(|&(_, off, ty)| (off, ty))
    }

    fn require(&self, name: &str) -> Result<(usize, PropType)> {
        self.offset_of(name).with_context(|| {
            format!(
                "PLY is missing the '{name}' property — not a 3D gaussian splatting file? \
                 Found properties: {}",
                self.props
                    .iter()
                    .map(|(n, ..)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}

/// Load a binary little-endian 3DGS PLY (the standard splat checkpoint format,
/// as written by Brush, the reference INRIA implementation, and most tools).
pub fn load_splat_ply(path: &Path) -> Result<SplatCloud> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Opening splat file {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let layout = parse_header(&mut reader)?;
    let x = layout.require("x")?;
    let y = layout.require("y")?;
    let z = layout.require("z")?;
    let opacity = layout.require("opacity")?;
    let s0 = layout.require("scale_0")?;
    let s1 = layout.require("scale_1")?;
    let s2 = layout.require("scale_2")?;
    let r0 = layout.require("rot_0")?;
    let r1 = layout.require("rot_1")?;
    let r2 = layout.require("rot_2")?;
    let r3 = layout.require("rot_3")?;
    let dc = ["f_dc_0", "f_dc_1", "f_dc_2"].map(|n| layout.offset_of(n));

    let mut data = vec![0u8; layout.stride * layout.count];
    reader
        .read_exact(&mut data)
        .context("PLY ended early — file truncated?")?;

    let f = |row: &[u8], (off, ty): (usize, PropType)| ty.read_f32(row, off);

    let n = layout.count;
    let mut cloud = SplatCloud {
        positions: Vec::with_capacity(n),
        scales: Vec::with_capacity(n),
        rotations: Vec::with_capacity(n),
        opacities: Vec::with_capacity(n),
        colors: Vec::with_capacity(n),
    };

    // Standard 3DGS stores logit opacities / log scales; some exporters bake
    // the activations in. Detect from the value range.
    let mut raw_opacity_min = f32::INFINITY;
    let mut raw_opacity_max = f32::NEG_INFINITY;
    for row in data.chunks_exact(layout.stride).take(10_000) {
        let o = f(row, opacity);
        raw_opacity_min = raw_opacity_min.min(o);
        raw_opacity_max = raw_opacity_max.max(o);
    }
    let opacity_is_linear = raw_opacity_min >= -0.01 && raw_opacity_max <= 1.01;
    if opacity_is_linear {
        log::info!("Splat PLY stores linear opacity/scale (activations baked in)");
    }

    for row in data.chunks_exact(layout.stride) {
        let pos = Vec3::new(f(row, x), f(row, y), f(row, z));
        if !pos.is_finite() {
            continue;
        }
        let raw_o = f(row, opacity);
        let o = if opacity_is_linear {
            raw_o
        } else {
            1.0 / (1.0 + (-raw_o).exp())
        };
        let raw_s = Vec3::new(f(row, s0), f(row, s1), f(row, s2));
        let scale = if opacity_is_linear {
            raw_s.abs()
        } else {
            Vec3::new(raw_s.x.exp(), raw_s.y.exp(), raw_s.z.exp())
        };
        // 3DGS quaternions are stored scalar-first: (w, x, y, z).
        let rot = Quat::from_xyzw(f(row, r1), f(row, r2), f(row, r3), f(row, r0));
        let rot = if rot.length_squared() > 1e-12 {
            rot.normalize()
        } else {
            Quat::IDENTITY
        };
        let color = match dc {
            [Some(c0), Some(c1), Some(c2)] => [
                (0.5 + SH_C0 * f(row, c0)).clamp(0.0, 1.0),
                (0.5 + SH_C0 * f(row, c1)).clamp(0.0, 1.0),
                (0.5 + SH_C0 * f(row, c2)).clamp(0.0, 1.0),
            ],
            _ => [0.6, 0.6, 0.6],
        };

        cloud.positions.push(pos);
        cloud.scales.push(scale);
        cloud.rotations.push(rot);
        cloud.opacities.push(o.clamp(0.0, 1.0));
        cloud.colors.push(color);
    }

    if cloud.is_empty() {
        bail!("No finite gaussians found in {}", path.display());
    }
    log::info!(
        "Loaded {} gaussians from {}",
        cloud.len(),
        path.display()
    );
    Ok(cloud)
}

fn parse_header(reader: &mut impl BufRead) -> Result<VertexLayout> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        bail!("Not a PLY file (missing 'ply' magic)");
    }

    let mut props = Vec::new();
    let mut stride = 0usize;
    let mut count = None;
    let mut in_vertex_element = false;
    let mut format_ok = false;

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            bail!("PLY header ended unexpectedly");
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["format", "binary_little_endian", _] => format_ok = true,
            ["format", other, _] => {
                bail!("Unsupported PLY format '{other}' (need binary_little_endian)")
            }
            ["comment", ..] | ["obj_info", ..] => {}
            ["element", "vertex", n] => {
                in_vertex_element = true;
                count = Some(n.parse::<usize>().context("Bad vertex count")?);
            }
            ["element", ..] => in_vertex_element = false,
            ["property", "list", ..] => {
                if in_vertex_element {
                    bail!("List properties on vertices are not supported");
                }
            }
            ["property", ty, name] if in_vertex_element => {
                let ty = PropType::parse(ty)
                    .with_context(|| format!("Unknown PLY property type '{ty}'"))?;
                props.push(((*name).to_owned(), stride, ty));
                stride += ty.size();
            }
            ["property", ..] => {}
            ["end_header"] => break,
            _ => {}
        }
    }

    if !format_ok {
        bail!("PLY header has no format line");
    }
    Ok(VertexLayout {
        props,
        stride,
        count: count.context("PLY has no vertex element")?,
    })
}
