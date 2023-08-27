#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>
#include <set>

#include "tools.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


class BaseApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;
	VkInstance instance;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	VkDebugUtilsMessengerEXT debug_messenger;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkRenderPass render_pass;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;
	std::vector<VkFramebuffer> swap_chain_framebuffers;
	VkCommandPool command_pool;
	std::vector<VkCommandBuffer> command_buffers;

	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;

	bool frame_buffer_resized = false;
	uint32_t current_frame = 0;
	

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
		app->frame_buffer_resized = true;
	}

	void initVulkan() {
		createVkInstance();
		setupDebugMessenger();
		createSurface();
		selectPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);
		

		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return; // Abandon this frame
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("Failed to acquire swap chain image");
		}

		vkResetFences(device, 1, &in_flight_fences[current_frame]);

		vkResetCommandBuffer(command_buffers[current_frame], 0);
		recordCommandbuffer(command_buffers[current_frame], image_index);

		VkSemaphore wait_semaphores[] = { image_available_semaphores[current_frame] };
		VkSemaphore signal_semaphores[] = { render_finished_semaphores[current_frame] };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = wait_semaphores,
			.pWaitDstStageMask = wait_stages,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffers[current_frame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signal_semaphores
		};
		
		VK_CHECK_RESULT(
			vkQueueSubmit(graphicsQueue, 1, &submit_info, in_flight_fences[current_frame])
		);

		VkSwapchainKHR swap_chains[] = { swapChain };
		VkPresentInfoKHR present_info{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signal_semaphores,
			.swapchainCount = 1,
			.pSwapchains = swap_chains,
			.pImageIndices = &image_index,
			.pResults = nullptr,
		};

		result = vkQueuePresentKHR(presentQueue, &present_info);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frame_buffer_resized) {
			frame_buffer_resized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("Failed to acquire swap chain image");
		}

		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void createSyncObjects() {
		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo sempaphore_create_info{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VkFenceCreateInfo fence_create_info{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT // Start the fence signaled for the first frame
		};

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VK_CHECK_RESULT(
				vkCreateSemaphore(device, &sempaphore_create_info, nullptr, &image_available_semaphores[i])
			);
			VK_CHECK_RESULT(
				vkCreateSemaphore(device, &sempaphore_create_info, nullptr, &render_finished_semaphores[i])
			);
			VK_CHECK_RESULT(
				vkCreateFence(device, &fence_create_info, nullptr, &in_flight_fences[i])
			);
		}

	}

	void recordCommandbuffer(VkCommandBuffer command_buffer, uint32_t image_index) {
		
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0,
			.pInheritanceInfo = nullptr,
		};

		VK_CHECK_RESULT(
			vkBeginCommandBuffer(command_buffer, &begin_info)
		);

		VkClearValue clear_color{ .color = {0.0f, 0.0f, 0.0f, 1.0f} };
		VkRenderPassBeginInfo render_pass_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = swap_chain_framebuffers[image_index],
			.renderArea = {
				.offset = {0, 0},
				.extent = swapChainExtent
			},
			.clearValueCount = 1,
			.pClearValues = &clear_color
		};

		vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(command_buffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(command_buffer, 0, 1, &scissor);

		vkCmdDraw(command_buffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(command_buffer);

		VK_CHECK_RESULT(
			vkEndCommandBuffer(command_buffer)
		);
	}

	void createCommandBuffers() {
		command_buffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo command_allocation_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = static_cast<uint32_t>(command_buffers.size())
		};

		VK_CHECK_RESULT(
			vkAllocateCommandBuffers(device, &command_allocation_info, command_buffers.data())
		);
	}

	void createCommandPool() {
		QueueFamilyIndices queue_family_indices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo pool_create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queue_family_indices.graphicsFamily.value()
		};

		VK_CHECK_RESULT(
			vkCreateCommandPool(device, &pool_create_info, nullptr, &command_pool)
		);
	}

	void createFramebuffers() {
		swap_chain_framebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			VkImageView attachments[] = {
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebuffer_info{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = render_pass,
				.attachmentCount = 1,
				.pAttachments = attachments,
				.width = swapChainExtent.width,
				.height = swapChainExtent.height,
				.layers = 1
			};

			VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebuffer_info, nullptr, &swap_chain_framebuffers[i]));
		}
	}

	void createRenderPass() {
		VkAttachmentDescription color_attachment{
			.format = swapChainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};

		VkAttachmentReference color_attachment_ref {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass_description{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref
		};

		VkSubpassDependency subpass_dependancy {
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		};

		VkRenderPassCreateInfo render_pass_create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &color_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass_description,
			.dependencyCount = 1,
			.pDependencies = &subpass_dependancy
		};

		if (vkCreateRenderPass(device, &render_pass_create_info, nullptr, &render_pass) != VK_SUCCESS) {
			throw std::runtime_error("Failed to crate render pass");
		}
	}

	void createGraphicsPipeline() {
		auto vert_shader_spv = readFile("shaders/vert.spv");
		auto frag_shader_spv = readFile("shaders/frag.spv");

		vertex_shader_module = createShaderModule(vert_shader_spv);
		fragment_shader_module = createShaderModule(frag_shader_spv);

		VkPipelineShaderStageCreateInfo vert_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main"  // Shader entry point
		};

		VkPipelineShaderStageCreateInfo frag_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main"  // Shader entry point
		};

		VkPipelineShaderStageCreateInfo shader_stages[] = { vert_create_info , frag_create_info };

		VkPipelineVertexInputStateCreateInfo vertex_input_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 0,
			.pVertexBindingDescriptions = nullptr,
			.vertexAttributeDescriptionCount = 0,
			.pVertexAttributeDescriptions = nullptr
		};

		VkPipelineInputAssemblyStateCreateInfo input_assembly{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = (float) swapChainExtent.width,
			.height = (float)swapChainExtent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};

		VkRect2D scissor{
			.offset = {.x = 0, .y = 0},
			.extent = swapChainExtent
		};

		std::vector<VkDynamicState> dynamic_states = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamic_state_crate_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
			.pDynamicStates = dynamic_states.data()
		};

		VkPipelineViewportStateCreateInfo viewport_state_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1
		};

		VkPipelineRasterizationStateCreateInfo rasterizer_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.depthBiasConstantFactor = 0.0f,
			.depthBiasClamp = 0.0f,
			.depthBiasSlopeFactor = 0.0f,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo multisample_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
			.minSampleShading = 1.0f,
			.pSampleMask = nullptr,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable = VK_FALSE,
		};

		VkPipelineColorBlendAttachmentState color_blend_attachment{
			.blendEnable = VK_FALSE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo color_blend_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &color_blend_attachment,
			.blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
		};

		VkPipelineLayoutCreateInfo pipeline_layout_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 0,
			.pSetLayouts = nullptr,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};

		if (vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to crate pipeline layout");
		}

		VkGraphicsPipelineCreateInfo pipeline_create_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shader_stages,
			.pVertexInputState = &vertex_input_info,
			.pInputAssemblyState = &input_assembly,
			.pViewportState = &viewport_state_create_info,
			.pRasterizationState = &rasterizer_create_info,
			.pMultisampleState = &multisample_create_info,
			.pDepthStencilState = nullptr,
			.pColorBlendState = &color_blend_create_info,
			.pDynamicState = &dynamic_state_crate_info,
			.layout = pipeline_layout,
			.renderPass = render_pass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex = -1,
		};

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create graphics pipeline");
		}

	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.size(),
			.pCode = reinterpret_cast<const uint32_t*>(code.data())
		};
		VkShaderModule shader_module;
		if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create shader module");
		}
		return shader_module;
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file");
		}
		size_t file_size = (size_t)file.tellg();
		std::vector<char> buffer(file_size);

		file.seekg(0);
		file.read(buffer.data(), file_size);
		file.close();
		return buffer;
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = swapChainImages[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = swapChainImageFormat,
				.components = {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				}
			};
			if (vkCreateImageView(device, &create_info, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create image views");
			}
		}
	}

	void cleanupSwapChain() {
		for (auto framebuffer : swap_chain_framebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createFramebuffers();
	}

	void createSwapChain() {
		SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
		VkPresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.presentModes);
		VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

		uint32_t min_image_count = swap_chain_support.capabilities.minImageCount;
		uint32_t max_image_count = swap_chain_support.capabilities.maxImageCount;

		std::cout << "Swap Chain support min: " << min_image_count << " | max: " << max_image_count << std::endl;
		uint32_t image_count = min_image_count + 1;
		if (image_count > max_image_count) {
			image_count = max_image_count;
		}

		swapChainImageFormat = surface_format.format;
		swapChainExtent = extent;

		VkSwapchainCreateInfoKHR create_info{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = image_count,
			.imageFormat = surface_format.format,
			.imageColorSpace = surface_format.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.preTransform = swap_chain_support.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = present_mode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE
		};

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queue_family_indices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices = queue_family_indices;
		}

		if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create swap chain");
		}
		vkGetSwapchainImagesKHR(device, swapChain, &image_count, nullptr);
		swapChainImages.resize(image_count);
		vkGetSwapchainImagesKHR(device, swapChain, &image_count, swapChainImages.data());
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create a window surface");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<uint32_t> unique_queue_families = { indices.graphicsFamily.value(), indices.presentFamily.value() };


		float queue_priority = 1.0f;
		for (uint32_t queue_family : unique_queue_families) {
	
			VkDeviceQueueCreateInfo queue_create_info{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = indices.graphicsFamily.value(),
				.queueCount = 1,
				.pQueuePriorities = &queue_priority
			};
			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features {};

		VkDeviceCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
			.pQueueCreateInfos = queue_create_infos.data(),
			.enabledLayerCount = 0,
			.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &device_features,
		};

		if (enableValidationLayers) {
			create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			create_info.ppEnabledLayerNames = validationLayers.data();
		}

		if (vkCreateDevice(physicalDevice, &create_info, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create logical device");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void selectPhysicalDevice() {
		uint32_t device_count = 0;

		vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
		if (device_count == 0) {
			throw std::runtime_error("Failed to find GPU with Vulkan Support");
		}

		std::vector<VkPhysicalDevice> devices(device_count);
		vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("Failed to find GPU with required features");
		}
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

		if (format_count != 0) {
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.presentModes.data());

		if (present_mode_count != 0) {
			details.presentModes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.presentModes.data());
		}
		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats) {

		for (const auto& available_format : available_formats) {
			if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return available_format;
			}
		}
		std::cerr << "Preferred swap surface format is not available, resorting to what is available." << std::endl;
		return available_formats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		// Prefer to use VK_PRESENT_MODE_MAILBOX_KHR if available 
		// Note: This is triple buffering, and may result in more GPU usage.
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			VkExtent2D actual_extent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height),
			};
			actual_extent.width = std::clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actual_extent.height = std::clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			
			return actual_extent;
		}
	}

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		// Logic to find queue family indices to populate struct with
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
				VkBool32 present_support = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
				if (present_support) {
					indices.presentFamily = i;
				}
			}
			i++;
		}
		return indices;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {

		VkPhysicalDeviceProperties device_properties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &device_properties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		std::cout << "Device Found: " << device_properties.deviceName << std::endl;

		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensions_supported = checkDeviceExtensionSupport(device);

		bool swap_chain_adequate = false;
		if (extensions_supported) {
			SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
			swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
		}

		return indices.isComplete() && extensions_supported && swap_chain_adequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extension_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
		std::vector<VkExtensionProperties> available_extensions(extension_count);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

		std::set<std::string> required_extensions(deviceExtensions.begin(), deviceExtensions.end());
		for (const auto& extension : available_extensions) {
			required_extensions.erase(extension.extensionName);
		}
		return required_extensions.empty();
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else {
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	VkDebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallBack;
		createInfo.pUserData = nullptr;
		createInfo.flags = 0;
		createInfo.pNext = nullptr;
		return createInfo;
	}


	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debug_messenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallBack(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
		void* p_user_data
	) {
		std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;
		return VK_FALSE;
	}



	std::vector<const char*> getVkExtensions() {
		uint32_t extensionCount = 0;
		const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensionCount);

		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + extensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		return extensions;
	}

	void createVkInstance() {
		if (enableValidationLayers && !checkVkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested but not available");
		}
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "WorldEngine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		std::vector<const char*> extensions = getVkExtensions();


		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);

			//createInfo.pNext = (VkDebugUtilsMessengerCreateIn07foEXT*) &createInfo;

			//pfnCreateDebugUtilsMessengerEXT(instance, &createInfo, NULL, &debug_messenger);
		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		if (result != VK_SUCCESS) {
			throw std::runtime_error("Failed to create VK instance");
		}
	}

	bool checkVkValidationLayerSupport() {

		std::cout << "Validating layer support" << std::endl;
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());


		for (const char* layerName : validationLayers) {
			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					std::cout << " - " << layerName << " : validated" << std::endl;
					layerFound = true;
					break;
				}
			}
			if (!layerFound){
				return false;
			}
		}
		return true;
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr) {
			func(instance, debugMessenger, pAllocator);
		}
	}

	void cleanup() {
		cleanupSwapChain();

		vkDestroyPipeline(device, graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyRenderPass(device, render_pass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
			vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
			vkDestroyFence(device, in_flight_fences[i], nullptr);
		}

		vkDestroyCommandPool(device, command_pool, nullptr);

		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
		}
		
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main() {
	BaseApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}