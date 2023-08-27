#!/bin/bash

$VULKAN_SDK/Bin/glslc.exe VulkanTutorial/shaders/src/shader.vert -o VulkanTutorial/shaders/vert.spv
$VULKAN_SDK/Bin/glslc.exe VulkanTutorial/shaders/src/shader.frag -o VulkanTutorial/shaders/frag.spv