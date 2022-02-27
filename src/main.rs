use cgmath::Deg;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Rad;
use cgmath::Vector3;

use image::GenericImageView;

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::single_layout_pool::SingleLayoutDescSet;
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::ImageDimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImmutableImage;
use vulkano::image::MipmapsCount;
use vulkano::image::SampleCount;
use vulkano::image::SwapchainImage;
use vulkano::image::view::ImageView;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::Version;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::sampler::Sampler;
use vulkano::swapchain::Capabilities;
use vulkano::swapchain::ColorSpace;
use vulkano::swapchain::CompositeAlpha;
use vulkano::swapchain::FullscreenExclusive;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SupportedPresentModes;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::Surface;
use vulkano::sync::GpuFuture;
use vulkano::sync::SharingMode;

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::Window;
use winit::window::WindowBuilder;

const TEXTURE_PATH: &str = "src/test.png";

#[derive(Copy, Clone)]
struct UniformBufferObject {
    // these are all used, but in the vertex shader. names need to match.
    #[allow(dead_code)]
    model: Matrix4<f32>,
    #[allow(dead_code)]
    view: Matrix4<f32>,
    #[allow(dead_code)]
    proj: Matrix4<f32>,
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
    colour: [f32; 3],
    tex: [f32; 2],
}

impl Vertex {
    fn new(pos: [f32; 3], colour: [f32; 3], tex: [f32; 2]) -> Self {
        Self { pos, colour, tex }
    }
}

vulkano::impl_vertex!(Vertex, pos, colour, tex);

struct QueueFamilyIndices {
    graphics_family: Option<usize>,
    present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self {
            graphics_family: None,
            present_family: None,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let instance = create_instance()?;

    let event_loop = EventLoop::new();
    let surface = create_surface(&event_loop, &instance);
    let physical_device_index = pick_physical_device(&instance, &surface)?;

    let (device, graphics_queue, present_queue) =
        create_logical_device(&instance, &surface, physical_device_index)?;

    let (swapchain, swapchain_images) = create_swapchain(
        &instance,
        &surface,
        physical_device_index,
        &device,
        &graphics_queue,
        &present_queue,
        None,
    )?;

    let depth_format = find_depth_format();
    let sample_count = find_sample_count();
    let render_pass =
        create_render_pass(&device, swapchain.format(), depth_format, sample_count)?;
    let pipeline = create_graphics_pipeline(&device, swapchain.dimensions(), &render_pass)?;

    let descriptor_sets_pool =
        create_descriptor_pool(&pipeline.layout().descriptor_set_layouts()[0].clone());

    let start_time = Instant::now();
    let uniform_buffers = create_uniform_buffers(
        &device,
        swapchain_images.len(),
        start_time,
        swapchain.dimensions(),
    )?;

    let texture_image = create_texture_image(&graphics_queue)?;
    let image_sampler = create_image_sampler(&device)?;
    let descriptor_sets = create_descriptor_sets(
        &descriptor_sets_pool,
        &uniform_buffers,
        &texture_image,
        &image_sampler,
    )?;

    Ok(())
}

fn create_descriptor_sets(
    pool: &Arc<Mutex<SingleLayoutDescSetPool>>,
    uniform_buffers: &[Arc<CpuAccessibleBuffer<UniformBufferObject>>],
    texture_image: &Arc<ImmutableImage>,
    image_sampler: &Arc<Sampler>,
) -> Result<Vec<Arc<SingleLayoutDescSet>>, Box<dyn Error>> {
    let mut ret: Vec<Arc<SingleLayoutDescSet>> = vec![];
    for uniform_buffer in uniform_buffers {
        let mut pool_locked = match pool.lock() {
            Ok(l) => l,
            Err(e) => return Err(e.to_string().into()),
        };

        ret.push(pool_locked.next([WriteDescriptorSet::image_view_sampler(
            0,
            ImageView::new(texture_image.clone())?,
            image_sampler.clone(),
        )])?);

        /*
        builder
            .add_buffer(uniform_buffer.clone())?
            .add_sampled_image(
                ImageView::new(texture_image.clone())?,
                image_sampler.clone(),
            )?;

        ret.push(builder.build()?)
        */
    }

    Ok(ret)
}

fn create_image_sampler(device: &Arc<Device>) -> Result<Arc<Sampler>, Box<dyn Error>> {
    Ok(Sampler::simple_repeat_linear(device.clone())?)
}

fn create_texture_image(graphics_queue: &Arc<Queue>) -> Result<Arc<ImmutableImage>, Box<dyn Error>> {
    let image = image::open(TEXTURE_PATH)?;

    let width = image.width();
    let height = image.height();

    let image_rgba = image.to_rgba8();

    let (image_view, future) = ImmutableImage::from_iter(
        image_rgba.into_raw().iter().cloned(),
        ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        },
        MipmapsCount::One,
        Format::R8G8B8A8_UNORM,
        graphics_queue.clone(),
    )?;

    future.flush()?;

    Ok(image_view)
}

fn create_uniform_buffers(
    device: &Arc<Device>,
    num_buffers: usize,
    start_time: Instant,
    dimensions: [u32; 2],
) -> Result<Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>, Box<dyn Error>> {
    let mut buffers = Vec::new();

    let dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let uniform_buffer = update_uniform_buffer(start_time, dimensions);

    for _ in 0..num_buffers {
        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer,
        )?;

        buffers.push(buffer);
    }

    Ok(buffers)
}

fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2]) -> UniformBufferObject {
    let duration = Instant::now().duration_since(start_time);
    let elapsed = (duration.as_secs() * 1000) + u64::from(duration.subsec_millis());

    let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.120)));

    let view = Matrix4::look_at_rh(
        Point3::new(2.0, 2.0, 2.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    );

    let mut proj = cgmath::perspective(
        Rad::from(Deg(45.0)),
        dimensions[0] / dimensions[1],
        0.1,
        10.0,
    );

    proj.y.y *= -1.0;

    UniformBufferObject { model, view, proj }
}

fn create_descriptor_pool(
    layout: &Arc<DescriptorSetLayout>,
) -> Arc<Mutex<SingleLayoutDescSetPool>> {
    Arc::new(Mutex::new(SingleLayoutDescSetPool::new(layout.clone())))
}

fn create_graphics_pipeline(
    device: &Arc<Device>,
    swapchain_extent: [u32; 2],
    render_pass: &Arc<RenderPass>,
) -> Result<Arc<GraphicsPipeline>, Box<dyn Error>> {
    mod vertex_shader {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shader.vert"
        }
    }

    mod fragment_shader {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shader.frag"
        }
    }

    let vert_shader_module = vertex_shader::load(device.clone())?;
    let frag_shader_module = fragment_shader::load(device.clone())?;

    let dimensions = [swapchain_extent[0] as f32, swapchain_extent[1] as f32];
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions,
        depth_range: 0.0..1.0,
    };

    let viewport_state = ViewportState::Fixed {
        data: vec![(viewport, Scissor::irrelevant())],
    };

    Ok(GraphicsPipeline::start()
       .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
       .vertex_shader(
           vert_shader_module.entry_point("main").unwrap(),
           (),
       )
       .viewport_state(viewport_state)
       .fragment_shader(
           frag_shader_module.entry_point("main").unwrap(),
           (),
       )
       .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
       .depth_stencil_state(DepthStencilState::simple_depth_test())
       .render_pass(
           Subpass::from(render_pass.clone(), 0).unwrap(),
       )
       .build(device.clone())?)
}

fn create_render_pass(
    device: &Arc<Device>,
    colour_format: Format,
    depth_format: Format,
    sample_count: SampleCount,
) -> Result<Arc<RenderPass>, Box<dyn Error>> {
    Ok(vulkano::single_pass_renderpass!(
        device.clone(),

        attachments: {
            multisample_color: {
                load: Clear,
                store: Store,
                format: colour_format,
                samples: sample_count,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::PresentSrc,
            },
            multisample_depth: {
                load: Clear,
                store: DontCare,
                format: depth_format,
                samples: sample_count,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            },
            resolve_color: {
                load: DontCare,
                store: Store,
                format: colour_format,
                samples: 1,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::PresentSrc,
            }
        },

        pass: {
            color: [multisample_color],
            depth_stencil: {multisample_depth},
            resolve: [resolve_color]
        }
    )?)
}

fn create_logical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
    physical_device_index: usize,
) -> Result<(Arc<Device>, Arc<Queue>, Arc<Queue>), Box<dyn Error>> {
    let physical_device = PhysicalDevice::from_index(&instance, physical_device_index)
        .unwrap();

    println!(
        "using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );
    println!("Vulkan version: {}", physical_device.api_version());
    println!(
        "driver version: {}",
        physical_device.properties().driver_version
    );

    let indices = find_queue_families(&surface, &physical_device)?;

    let families = [indices.graphics_family, indices.present_family];
    let unique_queue_families: HashSet<&Option<usize>> = HashSet::from_iter(families.iter());

    let queue_priority = 1.0;
    let queue_families: Result<Vec<_>, Box<dyn Error>> = unique_queue_families
        .iter()
        .map(|i| {
            Ok((
                physical_device
                    .queue_families()
                    .nth((**i).unwrap()).unwrap(),
                queue_priority,
            ))
        })
        .collect();

    let (device, mut queues) = Device::new(
        physical_device,
        physical_device.supported_features(),
        &device_extensions(),
        queue_families?,
    )?;

    let graphics_queue = queues
        .next()
        .unwrap();
    let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

    Ok((device, graphics_queue, present_queue))
}

fn find_sample_count() -> SampleCount {
    SampleCount::Sample8
}

fn find_depth_format() -> Format {
    Format::D16_UNORM
}

fn create_swapchain(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
    physical_device_index: usize,
    device: &Arc<Device>,
    graphics_queue: &Arc<Queue>,
    present_queue: &Arc<Queue>,
    old_swapchain: Option<Arc<Swapchain<Window>>>,
) -> Result<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>), Box<dyn Error>> {
    let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
    let capabilities = surface.capabilities(physical_device)?;

    let surface_format = choose_swap_surface_format(&capabilities.supported_formats);
    let present_mode = choose_swap_present_mode(capabilities.present_modes);
    let extent = choose_swap_extent(&capabilities);

    let mut image_count = capabilities.min_image_count + 1;
    if capabilities.max_image_count.is_some()
        && image_count > capabilities.max_image_count.unwrap()
    {
        image_count = capabilities.max_image_count.unwrap();
    }

    let image_usage = ImageUsage {
        color_attachment: true,
        transfer_destination: true,
        ..ImageUsage::none()
    };

    let indices = find_queue_families(&surface, &physical_device)?;

    let sharing: SharingMode = if indices.graphics_family == indices.present_family {
        graphics_queue.into()
    } else {
        vec![graphics_queue, present_queue].as_slice().into()
    };

    if let Some(old_swapchain) = old_swapchain {
        Ok(Swapchain::recreate(&old_swapchain)
           .num_images(image_count)
           .format(surface_format.0)
           .dimensions(extent)
           .layers(1)
           .usage(image_usage)
           .sharing_mode(sharing)
           .transform(capabilities.current_transform)
           .composite_alpha(CompositeAlpha::Opaque)
           .present_mode(present_mode)
           .fullscreen_exclusive(FullscreenExclusive::Allowed)
           .clipped(true)
           .color_space(surface_format.1)
           .build()?)
    } else {
        Ok(Swapchain::start(device.clone(), surface.clone())
           .num_images(image_count)
           .format(surface_format.0)
           .dimensions(extent)
           .layers(1)
           .usage(image_usage)
           .sharing_mode(sharing)
           .transform(capabilities.current_transform)
           .composite_alpha(CompositeAlpha::Opaque)
           .present_mode(present_mode)
           .fullscreen_exclusive(FullscreenExclusive::Allowed)
           .clipped(true)
           .color_space(surface_format.1)
           .build()?)
    }
}

fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
    if available_present_modes.mailbox {
        PresentMode::Mailbox
    } else {
        PresentMode::Fifo
    }
}

fn choose_swap_surface_format(
    available_formats: &[(Format, ColorSpace)],
) -> (Format, ColorSpace) {
    *available_formats
        .iter()
        .find(|(format, colour_space)| {
            *format == Format::B8G8R8A8_SRGB && *colour_space == ColorSpace::SrgbNonLinear
        })
        .unwrap_or_else(|| &available_formats[0])
}

fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
    if let Some(current_extent) = capabilities.current_extent {
        current_extent
    } else {
        let mut actual_extent = [800, 600];
        actual_extent[0] = capabilities.min_image_extent[0]
            .max(capabilities.max_image_extent[0].min(actual_extent[0]));
        actual_extent[1] = capabilities.min_image_extent[1]
            .max(capabilities.max_image_extent[1].min(actual_extent[1]));
        actual_extent
    }
}

fn pick_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
) -> Result<usize, Box<dyn Error>> {
    for (idx, dev) in PhysicalDevice::enumerate(instance).enumerate() {
        if is_device_suitable(surface, &dev)? {
            return Ok(idx);
        }
    }

    Err("Failed to find suitable GPU".into())
}

fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> Result<bool, Box<dyn Error>> {
    let indices = find_queue_families(surface, device)?;
    let extensions_supported = check_device_extension_support(device);

    let swapchain_adequate = if extensions_supported {
        let capabilities = surface.capabilities(*device)?;

        !capabilities.supported_formats.is_empty()
            && capabilities.present_modes.iter().next().is_some()
    } else {
        false
    };

    Ok(indices.is_complete() && extensions_supported && swapchain_adequate)
}

fn check_device_extension_support(device: &PhysicalDevice) -> bool {
    let available_extensions = device.supported_extensions();
    let device_extensions = device_extensions();
    available_extensions.intersection(&device_extensions) == device_extensions
}

fn find_queue_families(
    surface: &Arc<Surface<Window>>,
    device: &PhysicalDevice,
) -> Result<QueueFamilyIndices, Box<dyn Error>> {
    let mut indices = QueueFamilyIndices::new();
    for (i, queue_family) in device.queue_families().enumerate() {
        if queue_family.supports_graphics() {
            indices.graphics_family = Some(i);
        }

        if surface.is_supported(queue_family)? {
            indices.present_family = Some(i);
        }

        if indices.is_complete() {
            break;
        }
    }

    Ok(indices)
}

fn create_surface(
    event_loop: &EventLoop<()>,
    instance: &Arc<Instance>,
) -> Arc<Surface<Window>> {
    WindowBuilder::new()
        .with_title("test")
        .with_maximized(true)
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap()
}


fn create_instance() -> Result<Arc<Instance>, Box<dyn Error>> {
    let supported_extensions = InstanceExtensions::supported_by_core()?;
    println!("supported extensions: {:?}", supported_extensions);

    let app_info = ApplicationInfo {
        application_name: Some("test".into()),
        application_version: Some(Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse()?,
            minor: env!("CARGO_PKG_VERSION_MINOR").parse()?,
            patch: env!("CARGO_PKG_VERSION_PATCH").parse()?,
        }),
        engine_name: Some("No engine".into()),
        engine_version: Some(Version {
            major: 1,
            minor: 0,
            patch: 0,
        }),
    };

    let required_extensions = get_required_extensions();

    Ok(Instance::new(
        Some(&app_info),
        vulkano::Version::V1_2,
        &required_extensions,
        None,
    ).unwrap())
}

fn get_required_extensions() -> InstanceExtensions {
    vulkano_win::required_extensions()
}
