// Add `serde` for JSON serialization
use clap::{Parser, ValueEnum};
use colored::*;
use serde::Serialize;
use std::fmt;
use wgpu::Limits;

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
    Markdown,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
    output: OutputFormat,
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
        (*self as u64).pretty_format()
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

/// Converts a PCI vendor ID to a human-readable name.
fn vendor_to_string(vendor_id: u32) -> String {
    match vendor_id {
        0x1002 => "AMD".to_string(),
        0x10DE => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x13B5 => "ARM".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x1010 => "ImgTec".to_string(),
        _ => format!("Unknown (0x{:X})", vendor_id),
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
        "Max Push Constant Size:",
        limits.max_push_constant_size.pretty_format(),
    );
    print_row(
        MAX_KEY_LEN,
        "Max Inter-Stage Components:",
        limits.max_inter_stage_shader_components,
    );
}

fn print_markdown_output(info: &wgpu::AdapterInfo, limits: &Limits) {
    println!("## WGPU Adapter Information\n");
    println!("| Key | Value |");
    println!("|-----|-------|");
    println!("| Name | `{}` |", info.name);
    println!("| Backend | `{}` |", format!("{:?}", info.backend));
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
        &[
            (
                "Max Push Constant Size",
                format!("{} bytes", limits.max_push_constant_size),
            ),
            (
                "Max Inter-Stage Components",
                limits.max_inter_stage_shader_components.to_string(),
            ),
        ],
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

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
            println!("{}", serde_json::to_string_pretty(&report)?)
        }
        OutputFormat::Markdown => print_markdown_output(&info, &limits),
    }

    Ok(())
}
