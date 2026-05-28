#![allow(
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::match_bool,
    clippy::needless_for_each,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use dialoguer::{MultiSelect, theme::ColorfulTheme};
use serde::Serialize;
use std::{
    cmp::Ordering,
    fmt,
    sync::mpsc,
    time::{Duration, Instant},
};
use wgpu::Limits;

const MIB: u64 = 1024 * 1024;
const PAYLOAD_BENCHES: &[u64] = &[1, 10, 50];
const THROUGHPUT_MIB: u64 = 50;
const THROUGHPUT_RUNS: u32 = 10;
const PROFILE_TARGET_MIB: u64 = 256;
const BENCH_TIMESTAMP_ATTEMPTS: u8 = 3;

const RESNET50_SHAPES: &[MatmulShape] = &[
    MatmulShape::new(12544, 64, 147),
    MatmulShape::new(3136, 64, 64),
    MatmulShape::new(3136, 64, 576),
    MatmulShape::new(3136, 256, 64),
    MatmulShape::new(3136, 64, 256),
    MatmulShape::new(3136, 128, 256),
    MatmulShape::new(784, 128, 1152),
    MatmulShape::new(784, 512, 128),
    MatmulShape::new(784, 512, 256),
    MatmulShape::new(784, 128, 512),
    MatmulShape::new(784, 256, 512),
    MatmulShape::new(196, 256, 2304),
    MatmulShape::new(196, 1024, 256),
    MatmulShape::new(196, 1024, 512),
    MatmulShape::new(196, 256, 1024),
    MatmulShape::new(196, 512, 1024),
    MatmulShape::new(49, 512, 4608),
    MatmulShape::new(49, 2048, 512),
    MatmulShape::new(49, 2048, 1024),
    MatmulShape::new(49, 512, 2048),
];

const INCEPTION_V3_SHAPES: &[MatmulShape] = &[
    MatmulShape::new(22500, 32, 27),
    MatmulShape::new(22201, 32, 288),
    MatmulShape::new(22201, 64, 288),
    MatmulShape::new(5625, 80, 64),
    MatmulShape::new(5329, 192, 720),
    MatmulShape::new(1369, 64, 192),
    MatmulShape::new(324, 384, 2592),
    MatmulShape::new(81, 320, 1728),
    MatmulShape::new(100, 384, 4032),
    MatmulShape::new(9, 1001, 2048),
];

const TINY_SHAPES: &[MatmulShape] = &[
    MatmulShape::new(1, 1, 1),
    MatmulShape::new(4, 4, 4),
    MatmulShape::new(4, 4, 16),
    MatmulShape::new(1, 16, 16),
    MatmulShape::new(16, 16, 16),
];

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
    Markdown,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchProfile {
    Micro,
    Resnet50,
    Inceptionv3,
    Tiny,
}

impl BenchProfile {
    const fn all() -> &'static [Self] {
        &[Self::Micro, Self::Resnet50, Self::Inceptionv3, Self::Tiny]
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Micro => "micro",
            Self::Resnet50 => "resnet50",
            Self::Inceptionv3 => "inceptionv3",
            Self::Tiny => "tiny",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
    output: OutputFormat,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run GPU roundtrip benchmarks
    Bench {
        /// Benchmark profiles. Defaults to all profiles
        #[arg(value_enum)]
        profiles: Vec<BenchProfile>,

        /// Choose benchmark profiles interactively
        #[arg(short, long)]
        interactive: bool,

        /// List benchmark profiles
        #[arg(short, long)]
        list: bool,

        /// Sort each table by GPU runtime descending
        #[arg(long)]
        descending: bool,

        /// Sort each table by GPU runtime ascending
        #[arg(long)]
        ascending: bool,
    },
}

/// A combined struct for easy JSON serialization of all GPU info.
/// wgpu's `AdapterInfo` and `Limits` derive `Serialize` if the "serde" feature is enabled.
#[derive(Serialize)]
struct GpuReport<'a> {
    adapter_info: &'a wgpu::AdapterInfo,
    limits: &'a wgpu::Limits,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<&'a str>,
}

trait PrettyFormat {
    fn pretty_format(&self) -> String;
}

impl PrettyFormat for u32 {
    fn pretty_format(&self) -> String {
        u64::from(*self).pretty_format()
    }
}

impl PrettyFormat for u64 {
    fn pretty_format(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;
        const TB: u64 = GB * 1024;

        match *self {
            n if n >= TB => format!("{:.1} TB", n as f64 / TB as f64),
            n if n >= GB => format!("{:.1} GB", n as f64 / GB as f64),
            n if n >= MB => format!("{:.1} MB", n as f64 / MB as f64),
            n if n >= KB => format!("{:.1} KB", n as f64 / KB as f64),
            n => format!("{n} B"),
        }
    }
}

trait DurationFormat {
    fn ms(self) -> f64;
}

impl DurationFormat for Duration {
    fn ms(self) -> f64 {
        self.as_secs_f64() * 1000.0
    }
}

#[derive(Clone)]
struct BenchResult {
    label: String,
    payload_bytes: u64,
    runs: u32,
    cpu_roundtrip: Duration,
    gpu_time: Option<Duration>,
}

impl BenchResult {
    fn cpu_throughput_mib(&self) -> f64 {
        self.total_mib() / self.cpu_roundtrip.as_secs_f64()
    }

    fn gpu_throughput_mib(&self) -> Option<f64> {
        self.gpu_time
            .filter(|gpu_time| !gpu_time.is_zero())
            .map(|gpu_time| self.total_mib() / gpu_time.as_secs_f64())
    }

    fn total_mib(&self) -> f64 {
        (self.payload_bytes * u64::from(self.runs)) as f64 / MIB as f64
    }

    fn speedup_percent(&self) -> Option<f64> {
        self.gpu_time.map(|gpu_time| {
            ((self.cpu_roundtrip.as_secs_f64() / gpu_time.as_secs_f64()) - 1.0) * 100.0
        })
    }
}

#[derive(Clone, Copy)]
enum BenchSort {
    Ascending,
    Descending,
}

#[derive(Clone, Copy)]
struct MatmulShape {
    m: u64,
    n: u64,
    k: u64,
}

impl MatmulShape {
    const fn new(m: u64, n: u64, k: u64) -> Self {
        Self { m, n, k }
    }

    const fn payload_bytes(self) -> u64 {
        (self.m * self.k + self.k * self.n + self.m * self.n) * 4
    }

    fn label(self) -> String {
        format!("{}x{}x{}", self.m, self.n, self.k)
    }
}

/// Converts a PCI vendor ID to a human-readable name.
fn vendor_to_string(vendor_id: u32) -> String {
    match vendor_id {
        0x1002 => "AMD".to_string(),
        0x10DE => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x13B5 => "ARM".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x1010 => "ImgTec".to_string(),
        _ => format!("Unknown (0x{vendor_id:X})"),
    }
}

fn print_table_output(info: &wgpu::AdapterInfo, limits: &Limits) {
    let title = "WGPU Adapter Info & Device Limits".bold().underline();
    println!("{title}");

    /// All keys should fit in here...
    const MAX_KEY_LEN: usize = 30;

    fn print_row<T: fmt::Display>(max_key_len: usize, key: &str, value: T) {
        println!(
            "{: <max_key_len$} {}",
            key.cyan().bold(),
            value.to_string().yellow(),
            max_key_len = max_key_len
        );
    }

    // Adapter Information
    println!("\n{}", "Adapter Information".bold());
    print_row(MAX_KEY_LEN, "Name:", &info.name);
    print_row(MAX_KEY_LEN, "Backend:", format!("{:?}", info.backend));
    print_row(MAX_KEY_LEN, "Vendor:", vendor_to_string(info.vendor));
    print_row(MAX_KEY_LEN, "Device ID:", info.device);
    print_row(MAX_KEY_LEN, "Driver:", &info.driver);
    print_row(MAX_KEY_LEN, "Driver Info:", &info.driver_info);

    // Texture Limits
    println!("\n{}", "Texture Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max 1D Texture Size:",
        limits.max_texture_dimension_1d,
    );
    print_row(
        MAX_KEY_LEN,
        "Max 2D Texture Size:",
        limits.max_texture_dimension_2d,
    );
    print_row(
        MAX_KEY_LEN,
        "Max 3D Texture Size:",
        limits.max_texture_dimension_3d,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Array Layers:",
        limits.max_texture_array_layers,
    );

    // Binding Limits
    println!("\n{}", "Binding Limits".bold());
    print_row(MAX_KEY_LEN, "Max Bind Groups:", limits.max_bind_groups);
    print_row(
        MAX_KEY_LEN,
        "Max Bindings/Group:",
        limits.max_bindings_per_bind_group,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Dynamic Uniform Buffers:",
        limits.max_dynamic_uniform_buffers_per_pipeline_layout,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Dynamic Storage Buffers:",
        limits.max_dynamic_storage_buffers_per_pipeline_layout,
    );

    // Resource Limits
    println!("\n{}", "Resource Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max Sampled Textures:",
        limits.max_sampled_textures_per_shader_stage,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Samplers:",
        limits.max_samplers_per_shader_stage,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Storage Buffers:",
        limits.max_storage_buffers_per_shader_stage,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Storage Textures:",
        limits.max_storage_textures_per_shader_stage,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Uniform Buffers:",
        limits.max_uniform_buffers_per_shader_stage,
    );

    // Buffer Limits
    println!("\n{}", "Buffer Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max Uniform Buffer Size:",
        limits.max_uniform_buffer_binding_size.pretty_format(),
    );
    print_row(
        MAX_KEY_LEN,
        "Max Storage Buffer Size:",
        limits.max_storage_buffer_binding_size.pretty_format(),
    );
    print_row(
        MAX_KEY_LEN,
        "Min Uniform Alignment:",
        format!("{} bytes", limits.min_uniform_buffer_offset_alignment),
    );
    print_row(
        MAX_KEY_LEN,
        "Min Storage Alignment:",
        format!("{} bytes", limits.min_storage_buffer_offset_alignment),
    );

    // Vertex Limits
    println!("\n{}", "Vertex Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max Vertex Buffers:",
        limits.max_vertex_buffers,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Vertex Attributes:",
        limits.max_vertex_attributes,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Vertex Stride:",
        format!("{} bytes", limits.max_vertex_buffer_array_stride),
    );

    // Compute Limits
    println!("\n{}", "Compute Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max Workgroup Size X:",
        limits.max_compute_workgroup_size_x,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Workgroup Size Y:",
        limits.max_compute_workgroup_size_y,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Workgroup Size Z:",
        limits.max_compute_workgroup_size_z,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Workgroup Invocations:",
        limits.max_compute_invocations_per_workgroup,
    );
    print_row(
        MAX_KEY_LEN,
        "Max Workgroup Storage:",
        limits.max_compute_workgroup_storage_size.pretty_format(),
    );
    print_row(
        MAX_KEY_LEN,
        "Max Workgroups/Dimension:",
        limits.max_compute_workgroups_per_dimension,
    );

    // Misc Limits
    println!("\n{}", "Miscellaneous Limits".bold());
    print_row(
        MAX_KEY_LEN,
        "Max Inter-Stage Variables:",
        limits.max_inter_stage_shader_variables,
    );
}

fn print_markdown_output(info: &wgpu::AdapterInfo, limits: &Limits) {
    println!("## WGPU Adapter Information\n");
    println!("| Key | Value |");
    println!("|-----|-------|");
    println!("| Name | `{}` |", info.name);
    println!("| Backend | `{:?}` |", info.backend);
    println!("| Vendor | `{}` |", vendor_to_string(info.vendor));
    println!("| Device ID | `{}` |", info.device);
    println!("| Driver | `{}` |", info.driver);
    println!("| Driver Info | `{}` |", info.driver_info);

    println!("## WGPU Device Limits\n");

    fn print_section(section_title: &str, rows: &[(&str, String)]) {
        println!("### {section_title}\n");
        println!("| Key | Value |");
        println!("|-----|-------|");
        for (key, value) in rows {
            println!("| {key} | `{value}` |");
        }
        println!();
    }

    // Texture Limits
    print_section(
        "Texture Limits",
        &[
            (
                "Max 1D Texture Size",
                limits.max_texture_dimension_1d.to_string(),
            ),
            (
                "Max 2D Texture Size",
                limits.max_texture_dimension_2d.to_string(),
            ),
            (
                "Max 3D Texture Size",
                limits.max_texture_dimension_3d.to_string(),
            ),
            (
                "Max Array Layers",
                limits.max_texture_array_layers.to_string(),
            ),
        ],
    );

    // Binding Limits
    print_section(
        "Binding Limits",
        &[
            ("Max Bind Groups", limits.max_bind_groups.to_string()),
            (
                "Max Bindings/Group",
                limits.max_bindings_per_bind_group.to_string(),
            ),
            (
                "Max Dynamic Uniform Buffers",
                limits
                    .max_dynamic_uniform_buffers_per_pipeline_layout
                    .to_string(),
            ),
            (
                "Max Dynamic Storage Buffers",
                limits
                    .max_dynamic_storage_buffers_per_pipeline_layout
                    .to_string(),
            ),
        ],
    );

    // Resource Limits
    print_section(
        "Resource Limits",
        &[
            (
                "Max Sampled Textures",
                limits.max_sampled_textures_per_shader_stage.to_string(),
            ),
            (
                "Max Samplers",
                limits.max_samplers_per_shader_stage.to_string(),
            ),
            (
                "Max Storage Buffers",
                limits.max_storage_buffers_per_shader_stage.to_string(),
            ),
            (
                "Max Storage Textures",
                limits.max_storage_textures_per_shader_stage.to_string(),
            ),
            (
                "Max Uniform Buffers",
                limits.max_uniform_buffers_per_shader_stage.to_string(),
            ),
        ],
    );

    // Buffer Limits
    print_section(
        "Buffer Limits",
        &[
            (
                "Max Uniform Buffer Size",
                limits.max_uniform_buffer_binding_size.pretty_format(),
            ),
            (
                "Max Storage Buffer Size",
                limits.max_storage_buffer_binding_size.pretty_format(),
            ),
            (
                "Min Uniform Alignment",
                format!("{} bytes", limits.min_uniform_buffer_offset_alignment),
            ),
            (
                "Min Storage Alignment",
                format!("{} bytes", limits.min_storage_buffer_offset_alignment),
            ),
        ],
    );

    // Vertex Limits
    print_section(
        "Vertex Limits",
        &[
            ("Max Vertex Buffers", limits.max_vertex_buffers.to_string()),
            (
                "Max Vertex Attributes",
                limits.max_vertex_attributes.to_string(),
            ),
            (
                "Max Vertex Stride",
                format!("{} bytes", limits.max_vertex_buffer_array_stride),
            ),
        ],
    );

    // Compute Limits
    print_section(
        "Compute Limits",
        &[
            (
                "Max Workgroup Size X",
                limits.max_compute_workgroup_size_x.to_string(),
            ),
            (
                "Max Workgroup Size Y",
                limits.max_compute_workgroup_size_y.to_string(),
            ),
            (
                "Max Workgroup Size Z",
                limits.max_compute_workgroup_size_z.to_string(),
            ),
            (
                "Max Workgroup Invocations",
                limits.max_compute_invocations_per_workgroup.to_string(),
            ),
            (
                "Max Workgroup Storage",
                limits.max_compute_workgroup_storage_size.pretty_format(),
            ),
            (
                "Max Workgroups/Dimension",
                limits.max_compute_workgroups_per_dimension.to_string(),
            ),
        ],
    );

    // Misc Limits
    print_section(
        "Miscellaneous Limits",
        &[(
            "Max Inter-Stage Variables",
            limits.max_inter_stage_shader_variables.to_string(),
        )],
    );
}

async fn request_bench_device() -> Result<
    (wgpu::AdapterInfo, wgpu::Features, wgpu::Device, wgpu::Queue),
    Box<dyn std::error::Error>,
> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpucheck bench device"),
            required_features: features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;

    Ok((adapter.get_info(), adapter.features(), device, queue))
}

fn make_source_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpucheck bench source"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });

    {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        let data = (0..size)
            .map(|index| index.to_le_bytes()[0].wrapping_mul(31).wrapping_add(7))
            .collect::<Vec<_>>();
        view.copy_from_slice(&data);
    }
    buffer.unmap();

    buffer
}

fn verify_roundtrip(readback: &wgpu::Buffer, size: u64) -> Result<(), Box<dyn std::error::Error>> {
    let view = readback.slice(..).get_mapped_range();
    let first = view.first().copied();
    let last = view.last().copied();
    drop(view);
    readback.unmap();

    match (first, last) {
        (Some(7), Some(byte))
            if byte == (size - 1).to_le_bytes()[0].wrapping_mul(31).wrapping_add(7) =>
        {
            Ok(())
        }
        _ => Err("GPU roundtrip payload check failed".into()),
    }
}

fn run_copy_bench(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    payload_bytes: u64,
    runs: u32,
    label: &str,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let mut last_result = None;

    for _ in 0..BENCH_TIMESTAMP_ATTEMPTS {
        let result = run_copy_bench_once(device, queue, payload_bytes, runs, label)?;
        if result.gpu_time.is_some() {
            return Ok(result);
        }
        last_result = Some(result);
    }

    last_result.ok_or_else(|| "bench did not run".into())
}

fn run_copy_bench_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    payload_bytes: u64,
    runs: u32,
    label: &str,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let source = make_source_buffer(device, payload_bytes);
    let middle = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpucheck bench middle"),
        size: payload_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpucheck bench output"),
        size: payload_bytes,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpucheck bench readback"),
        size: payload_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let word_count = payload_bytes / 4;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgpucheck bench shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "
struct Words {{
    data: array<u32>,
}};

@group(0) @binding(0) var<storage, read> source: Words;
@group(0) @binding(1) var<storage, read_write> middle: Words;
@group(0) @binding(2) var<storage, read_write> output: Words;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let index = id.x;
    if (index >= {word_count}u) {{
        return;
    }}

    let mixed = source.data[index] ^ 0xa5a5a5a5u;
    middle.data[index] = mixed;
    storageBarrier();
    output.data[index] = middle.data[index] ^ 0xa5a5a5a5u;
}}
"
            )
            .into(),
        ),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wgpucheck bench bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wgpucheck bench pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wgpucheck bench pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpucheck bench bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: middle.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let timestamp_query_set = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("wgpucheck bench timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
    let timestamp_resolve = timestamp_query_set.as_ref().map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpucheck bench timestamp resolve"),
            size: u64::from(wgpu::QUERY_SIZE) * 2,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let timestamp_readback = timestamp_query_set.as_ref().map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpucheck bench timestamp readback"),
            size: u64::from(wgpu::QUERY_SIZE) * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });

    let start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpucheck bench encoder"),
    });

    {
        let timestamp_writes =
            timestamp_query_set
                .as_ref()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpucheck bench compute"),
            timestamp_writes,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = u32::try_from(word_count.div_ceil(256))?;
        (0..runs).for_each(|_| pass.dispatch_workgroups(workgroups, 1, 1));
    }

    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, payload_bytes);

    if let (Some(query_set), Some(resolve), Some(timestamp_readback)) = (
        &timestamp_query_set,
        &timestamp_resolve,
        &timestamp_readback,
    ) {
        encoder.resolve_query_set(query_set, 0..2, resolve, 0);
        encoder.copy_buffer_to_buffer(
            resolve,
            0,
            timestamp_readback,
            0,
            u64::from(wgpu::QUERY_SIZE) * 2,
        );
    }

    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;
    let cpu_roundtrip = start.elapsed();

    verify_roundtrip(&readback, payload_bytes)?;

    let gpu_time = timestamp_readback
        .as_ref()
        .map(|buffer| read_timestamp_duration(device, queue, buffer))
        .transpose()?
        .flatten()
        .filter(|duration| *duration <= cpu_roundtrip);

    Ok(BenchResult {
        label: label.to_string(),
        payload_bytes,
        runs,
        cpu_roundtrip,
        gpu_time,
    })
}

fn read_timestamp_duration(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Result<Option<Duration>, Box<dyn std::error::Error>> {
    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let view = buffer.slice(..).get_mapped_range();
    let start = u64::from_le_bytes(view[0..wgpu::QUERY_SIZE as usize].try_into()?);
    let end = u64::from_le_bytes(
        view[wgpu::QUERY_SIZE as usize..(wgpu::QUERY_SIZE as usize * 2)].try_into()?,
    );
    drop(view);
    buffer.unmap();

    let ticks = end.wrapping_sub(start);
    match ticks {
        0 => Ok(None),
        ticks => {
            let nanoseconds = ticks as f64 * f64::from(queue.get_timestamp_period());
            Ok(Some(Duration::from_secs_f64(nanoseconds / 1_000_000_000.0)))
        }
    }
}

fn print_bench_results(
    profile: BenchProfile,
    adapter_info: &wgpu::AdapterInfo,
    adapter_features: wgpu::Features,
    device_features: wgpu::Features,
    results: &[BenchResult],
) {
    println!(
        "{} {}",
        "WGPU GPU Roundtrip Bench".bold().underline(),
        profile.name().yellow().bold()
    );
    println!(
        "{} {}",
        "Adapter:".cyan().bold(),
        adapter_info.name.yellow()
    );
    println!(
        "{} adapter={} device={}",
        "Timestamps:".cyan().bold(),
        timestamp_feature_text(adapter_features),
        timestamp_feature_text(device_features)
    );
    println!();
    println!(
        "{} {} {} {} {} {} {} {} {}",
        format!("{:<12}", "Profile").cyan().bold(),
        format!("{:<20}", "Bench").cyan().bold(),
        format!("{:>10}", "Payload").cyan().bold(),
        format!("{:>6}", "Runs").cyan().bold(),
        format!("{:>14}", "CPU ms").cyan().bold(),
        format!("{:>14}", "GPU ms").cyan().bold(),
        format!("{:>12}", "Speedup %").cyan().bold(),
        format!("{:>16}", "CPU MiB/s").cyan().bold(),
        format!("{:>16}", "GPU MiB/s").cyan().bold()
    );
    println!("{}", "-".repeat(129).bright_black());
    results.iter().for_each(|result| {
        let gpu_ms = result.gpu_time.map_or_else(
            || format!("{:>14}", "n/a").red().to_string(),
            |duration| format!("{:>14.6}", duration.ms()).green().to_string(),
        );
        let gpu_throughput = result.gpu_throughput_mib().map_or_else(
            || format!("{:>16}", "n/a").red().to_string(),
            |throughput| format!("{throughput:>16.1}").green().to_string(),
        );
        let speedup = result.speedup_percent().map_or_else(
            || format!("{:>12}", "n/a").red().to_string(),
            |speedup| format!("{speedup:>+11.1}%").green().to_string(),
        );

        println!(
            "{} {} {} {} {} {} {} {} {}",
            format!("{:<12}", profile.name()).cyan(),
            format!("{:<20}", result.label).cyan(),
            format!("{:>10}", result.payload_bytes.pretty_format()).yellow(),
            format!("{:>6}", result.runs).yellow(),
            format!("{:>14.3}", result.cpu_roundtrip.ms()).yellow(),
            gpu_ms,
            speedup,
            format!("{:>16.1}", result.cpu_throughput_mib()).yellow(),
            gpu_throughput
        );
    });
}

fn timestamp_feature_text(features: wgpu::Features) -> &'static str {
    match (
        features.contains(wgpu::Features::TIMESTAMP_QUERY),
        features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
    ) {
        (true, true) => "query+encoder",
        (true, false) => "query-only",
        (false, _) => "none",
    }
}

fn profile_runs(payload_bytes: u64) -> u32 {
    let target_bytes = PROFILE_TARGET_MIB * MIB;
    let runs = target_bytes.div_ceil(payload_bytes).max(4);
    u32::try_from(runs).unwrap_or(u32::MAX)
}

fn push_shape_benches(
    results: &mut Vec<BenchResult>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shapes: &[MatmulShape],
) -> Result<(), Box<dyn std::error::Error>> {
    for shape in shapes {
        let label = shape.label();
        results.push(run_copy_bench(
            device,
            queue,
            shape.payload_bytes(),
            profile_runs(shape.payload_bytes()),
            &label,
        )?);
    }

    Ok(())
}

fn run_profile_benchmarks(
    profile: BenchProfile,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<Vec<BenchResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();

    match profile {
        BenchProfile::Micro => {
            for payload_mib in PAYLOAD_BENCHES {
                let label = format!("{payload_mib} MiB roundtrip");
                results.push(run_copy_bench(device, queue, payload_mib * MIB, 1, &label)?);
            }

            let label = format!("{THROUGHPUT_MIB} MiB throughput x {THROUGHPUT_RUNS}");
            results.push(run_copy_bench(
                device,
                queue,
                THROUGHPUT_MIB * MIB,
                THROUGHPUT_RUNS,
                &label,
            )?);
        }
        BenchProfile::Resnet50 => {
            push_shape_benches(&mut results, device, queue, RESNET50_SHAPES)?;
        }
        BenchProfile::Inceptionv3 => {
            push_shape_benches(&mut results, device, queue, INCEPTION_V3_SHAPES)?;
        }
        BenchProfile::Tiny => {
            push_shape_benches(&mut results, device, queue, TINY_SHAPES)?;
        }
    }

    Ok(results)
}

fn sort_bench_results(results: &mut [BenchResult], sort: Option<BenchSort>) {
    let Some(sort) = sort else {
        return;
    };

    results.sort_by(|left, right| match (left.gpu_time, right.gpu_time) {
        (Some(left), Some(right)) => match sort {
            BenchSort::Ascending => left.cmp(&right),
            BenchSort::Descending => right.cmp(&left),
        },
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
}

fn list_bench_profiles() {
    println!("{}", "Benchmark profiles".bold().underline());
    BenchProfile::all()
        .iter()
        .for_each(|profile| println!("{}", profile.name().yellow()));
}

fn choose_bench_profiles() -> Result<Vec<BenchProfile>, Box<dyn std::error::Error>> {
    let profiles = BenchProfile::all();
    let items = profiles
        .iter()
        .map(|profile| profile.name())
        .collect::<Vec<_>>();
    let selection = MultiSelect::with_theme(&ColorfulTheme::default())
        .with_prompt("Select benchmark profiles")
        .items(&items)
        .interact()?;

    match selection.is_empty() {
        true => Err("No benchmark profiles selected".into()),
        false => Ok(selection.into_iter().map(|index| profiles[index]).collect()),
    }
}

fn bench_profiles_or_default(
    profiles: Vec<BenchProfile>,
    interactive: bool,
) -> Result<Vec<BenchProfile>, Box<dyn std::error::Error>> {
    match (interactive, profiles.is_empty()) {
        (true, _) => choose_bench_profiles(),
        (false, true) => Ok(BenchProfile::all().to_vec()),
        (false, false) => Ok(profiles),
    }
}

fn run_benchmarks(
    profiles: Vec<BenchProfile>,
    sort: Option<BenchSort>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (adapter_info, adapter_features, device, queue) =
        pollster::block_on(request_bench_device())?;
    let device_features = device.features();

    for profile in profiles {
        let mut results = run_profile_benchmarks(profile, &device, &queue)?;
        sort_bench_results(&mut results, sort);
        print_bench_results(
            profile,
            &adapter_info,
            adapter_features,
            device_features,
            &results,
        );
        println!();
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    if let Some(Command::Bench {
        profiles,
        interactive,
        list,
        descending,
        ascending,
    }) = args.command
    {
        if list {
            list_bench_profiles();
            return Ok(());
        }

        if ascending && descending {
            return Err("Use only one of --ascending or --descending".into());
        }

        let sort = match (ascending, descending) {
            (true, false) => Some(BenchSort::Ascending),
            (false, true) => Some(BenchSort::Descending),
            _ => None,
        };

        return run_benchmarks(bench_profiles_or_default(profiles, interactive)?, sort);
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
    })?;

    let info = adapter.get_info();
    let limits = adapter.limits();

    match args.output {
        OutputFormat::Table => print_table_output(&info, &limits),
        OutputFormat::Json => {
            let report = GpuReport {
                adapter_info: &info,
                limits: &limits,
                notes: None,
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        OutputFormat::Markdown => print_markdown_output(&info, &limits),
    }

    Ok(())
}
