"""
Metal GPU acceleration for spatial calculations on M2 MacBooks.
Handles efficient resource distribution, distance calculations, and environment processing.
"""

import sys
import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import warnings
import platform

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIG, METAL_CONFIG

# Check if we're on a Mac with Apple Silicon
is_apple_silicon = (
        platform.system() == "Darwin" and
        platform.processor() == "arm" and
        platform.machine() == "arm64"
)

# Try to import Metal packages
try:
    if is_apple_silicon:
        import Metal
        import Foundation
        # Print debugging info
        device = Metal.MTLCreateSystemDefaultDevice()
        if device:
            print(f"Metal initialized successfully on: {device.name()}")
            HAS_METAL = True
        else:
            print("Metal device creation failed")
            HAS_METAL = False
    else:
        HAS_METAL = False
except ImportError as e:
    HAS_METAL = False
    warnings.warn(f"Metal libraries not found: {e}. GPU acceleration disabled.")


class MetalCompute:
    """Handles GPU-accelerated computations using Metal on M2 Macs."""

    def __init__(self):
        """Initialize Metal compute resources."""
        self.has_metal = HAS_METAL

        if not self.has_metal:
            warnings.warn("Metal not available. Falling back to CPU.")
            return

        # Initialize Metal resources
        self._setup_metal()

        # Compile Metal functions
        self._compile_kernels()

    def _setup_metal(self):
        """Set up Metal device, command queue, and library with improved error handling."""
        if not self.has_metal:
            return

        try:
            # Get default Metal device
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                warnings.warn("No Metal device found. Falling back to CPU.")
                self.has_metal = False
                return

            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None:
                warnings.warn("Failed to create Metal command queue. Falling back to CPU.")
                self.has_metal = False
                return

            # Update config
            METAL_CONFIG["device"] = self.device
            METAL_CONFIG["command_queue"] = self.command_queue

            # Print available memory info if possible
            try:
                print(f"Metal device: {self.device.name()}")
                if hasattr(self.device, "recommendedMaxWorkingSetSize"):
                    max_mem = self.device.recommendedMaxWorkingSetSize() / (1024 * 1024)
                    print(f"Available Metal memory: {max_mem:.2f} MB")
            except Exception as e:
                print(f"Could not retrieve Metal device info: {e}")

        except Exception as e:
            warnings.warn(f"Error initializing Metal: {e}. Falling back to CPU.")
            self.has_metal = False

    def _compile_kernels(self):
        """Compile Metal kernel functions with proper error handling."""
        if not self.has_metal:
            return

        # Metal shader code for spatial calculations
        metal_code = """
        #include <metal_stdlib>
        using namespace metal;

        // Kernel for calculating distances between agents
        kernel void calculate_distances(
            device const float2 *positions [[ buffer(0) ]],
            device float *distances [[ buffer(1) ]],
            constant uint &num_agents [[ buffer(2) ]],
            uint2 gid [[ thread_position_in_grid ]]
        ) {
            const uint i = gid.x;
            const uint j = gid.y;

            // Only calculate upper triangle to avoid redundancy
            if (i < num_agents && j < num_agents && i < j) {
                float2 pos_i = positions[i];
                float2 pos_j = positions[j];

                float2 diff = pos_i - pos_j;
                float dist = sqrt(diff.x * diff.x + diff.y * diff.y);

                // Store in linear array
                uint index = i * num_agents + j;
                distances[index] = dist;
            }
        }

        // Kernel for calculating resource influence on environment
        kernel void calculate_resource_influence(
            device const float2 *resource_positions [[ buffer(0) ]],
            device const float *resource_values [[ buffer(1) ]],
            device float *influence_map [[ buffer(2) ]],
            constant uint2 &grid_size [[ buffer(3) ]],
            constant uint &num_resources [[ buffer(4) ]],
            constant float &influence_radius [[ buffer(5) ]],
            uint2 gid [[ thread_position_in_grid ]]
        ) {
            // Calculate position in the grid
            const uint x = gid.x;
            const uint y = gid.y;

            if (x < grid_size.x && y < grid_size.y) {
                float2 pos = float2(x, y);
                float influence = 0.0f;

                // Accumulate influence from each resource
                for (uint i = 0; i < num_resources; i++) {
                    float2 resource_pos = resource_positions[i];
                    float value = resource_values[i];

                    float2 diff = pos - resource_pos;
                    float dist = sqrt(diff.x * diff.x + diff.y * diff.y);

                    // Add influence based on distance and value
                    if (dist < influence_radius) {
                        float factor = 1.0f - (dist / influence_radius);
                        influence += value * factor * factor;
                    }
                }

                // Store result
                uint index = y * grid_size.x + x;
                influence_map[index] = influence;
            }
        }

        // Kernel for spatial partitioning grid updates - FIXED for atomic operations
        kernel void update_partition_grid(
            device const float2 *positions [[ buffer(0) ]],
            device atomic_uint *grid_counts [[ buffer(1) ]],  // FIXED: Changed to atomic_uint
            device uint *grid_indices [[ buffer(2) ]],
            constant uint2 &grid_size [[ buffer(3) ]],
            constant float2 &cell_size [[ buffer(4) ]],
            constant uint &num_entities [[ buffer(5) ]],
            uint gid [[ thread_position_in_grid ]]
        ) {
            const uint entity_id = gid;

            if (entity_id < num_entities) {
                float2 pos = positions[entity_id];

                // Calculate grid cell
                uint cell_x = min(uint(pos.x / cell_size.x), grid_size.x - 1);
                uint cell_y = min(uint(pos.y / cell_size.y), grid_size.y - 1);

                // Calculate linear index
                uint cell_index = cell_y * grid_size.x + cell_x;

                // Atomically increment counter and get index - now using proper atomic type
                uint index = atomic_fetch_add_explicit(&grid_counts[cell_index], 1u, memory_order_relaxed);

                // Store entity ID in grid
                uint storage_index = cell_index * num_entities + index;
                if (storage_index < grid_size.x * grid_size.y * num_entities) {
                    grid_indices[storage_index] = entity_id;
                }
            }
        }
        """

        try:
            # Create Metal library
            options = Metal.MTLCompileOptions.alloc().init()
            source = Foundation.NSString.alloc().initWithUTF8String_(metal_code.encode('utf-8'))

            # PyObjC methods with error parameters return tuples (result, error)
            result = self.device.newLibraryWithSource_options_error_(source, options, None)
            library = result[0]  # First element is the library
            error = result[1]  # Second element is the error (if any)

            if library is None:
                error_msg = error.localizedDescription() if error else "Unknown error"
                warnings.warn(f"Failed to compile Metal library: {error_msg}")
                self.has_metal = False
                return

            print("Metal shaders compiled successfully")

            # Create function objects
            self.calculate_distances_function = library.newFunctionWithName_("calculate_distances")
            self.resource_influence_function = library.newFunctionWithName_("calculate_resource_influence")
            self.update_partition_grid_function = library.newFunctionWithName_("update_partition_grid")

            if (self.calculate_distances_function is None or
                    self.resource_influence_function is None or
                    self.update_partition_grid_function is None):
                warnings.warn("Failed to get Metal functions from library")
                self.has_metal = False
                return

            # Create compute pipeline states - properly unpack tuples
            result = self.device.newComputePipelineStateWithFunction_error_(
                self.calculate_distances_function, None)
            self.distance_pipeline = result[0]
            error = result[1]

            if self.distance_pipeline is None:
                error_msg = error.localizedDescription() if error else "Unknown error"
                warnings.warn(f"Failed to create distance calculation pipeline: {error_msg}")
                self.has_metal = False
                return

            result = self.device.newComputePipelineStateWithFunction_error_(
                self.resource_influence_function, None)
            self.resource_pipeline = result[0]
            error = result[1]

            if self.resource_pipeline is None:
                error_msg = error.localizedDescription() if error else "Unknown error"
                warnings.warn(f"Failed to create resource influence pipeline: {error_msg}")
                self.has_metal = False
                return

            result = self.device.newComputePipelineStateWithFunction_error_(
                self.update_partition_grid_function, None)
            self.partition_pipeline = result[0]
            error = result[1]

            if self.partition_pipeline is None:
                error_msg = error.localizedDescription() if error else "Unknown error"
                warnings.warn(f"Failed to create partition pipeline: {error_msg}")
                self.has_metal = False
                return

            # Update config
            METAL_CONFIG["library"] = library

            print("Metal GPU acceleration successfully initialized")

        except Exception as e:
            warnings.warn(f"Error during Metal setup: {str(e)}")
            self.has_metal = False

    def calculate_distances(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between all positions using GPU.

        Args:
            positions: Array of (x, y) positions with shape (n, 2)

        Returns:
            Distance matrix with shape (n, n)
        """
        if not self.has_metal or not HAS_METAL:
            # Fallback to CPU implementation
            return self._calculate_distances_cpu(positions)

        num_agents = positions.shape[0]

        # Create input buffer
        positions_buffer = self.device.newBufferWithBytes_length_options_(
            positions.astype(np.float32).tobytes(),
            positions.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create output buffer
        distance_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
        distances_buffer = self.device.newBufferWithBytes_length_options_(
            distance_matrix.tobytes(),
            distance_matrix.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create parameter buffer
        num_agents_array = np.array([num_agents], dtype=np.uint32)
        params_buffer = self.device.newBufferWithBytes_length_options_(
            num_agents_array.tobytes(),
            num_agents_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Configure pipeline
        compute_encoder.setComputePipelineState_(self.distance_pipeline)
        compute_encoder.setBuffer_offset_atIndex_(positions_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(distances_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        # Configure grid and threadgroups
        threads_per_group = min(self.distance_pipeline.maxTotalThreadsPerThreadgroup(), 16)
        threadgroup_size = Metal.MTLSizeMake(threads_per_group, threads_per_group, 1)
        grid_size = Metal.MTLSizeMake(num_agents, num_agents, 1)

        # Dispatch
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, threadgroup_size)
        compute_encoder.endEncoding()

        # Execute and wait with error handling
        try:
            # Safety check for buffers
            if positions_buffer is None or distances_buffer is None:
                warnings.warn("Invalid Metal buffers, falling back to CPU implementation")
                return self._calculate_distances_cpu(positions)

            # Execute and wait with error catching
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            # Check for errors
            if hasattr(command_buffer, 'error') and command_buffer.error():
                warnings.warn(f"Metal execution error: {command_buffer.error().localizedDescription()}")
                return self._calculate_distances_cpu(positions)
        except Exception as e:
            warnings.warn(f"Exception during Metal execution: {e}")
            return self._calculate_distances_cpu(positions)

        # Copy results back to numpy
        result_ptr = distances_buffer.contents()
        output = np.ctypeslib.as_array(
            (np.ctypeslib.ct.c_float * (num_agents * num_agents)).from_address(int(result_ptr)),
            shape=(num_agents, num_agents)
        ).copy()

        # Make matrix symmetric
        output = output + output.T

        return output

    def _calculate_distances_cpu(self, positions: np.ndarray) -> np.ndarray:
        """CPU fallback for distance calculations."""
        num_agents = positions.shape[0]
        distances = np.zeros((num_agents, num_agents), dtype=np.float32)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distances[i, j] = distances[j, i] = np.sqrt(dx * dx + dy * dy)

        return distances

    def calculate_resource_influence(
            self,
            resource_positions: np.ndarray,
            resource_values: np.ndarray,
            grid_size: Tuple[int, int],
            influence_radius: float = 50.0
    ) -> np.ndarray:
        """
        Calculate resource influence across the environment grid.

        Args:
            resource_positions: Array of resource positions with shape (n, 2)
            resource_values: Array of resource values with shape (n,)
            grid_size: Size of the grid as (width, height)
            influence_radius: Maximum distance of resource influence

        Returns:
            Influence map with shape grid_size
        """
        if not self.has_metal or not HAS_METAL:
            # Fallback to CPU implementation
            return self._calculate_resource_influence_cpu(
                resource_positions, resource_values, grid_size, influence_radius
            )

        num_resources = resource_positions.shape[0]

        # Create input buffers
        positions_buffer = self.device.newBufferWithBytes_length_options_(
            resource_positions.astype(np.float32).tobytes(),
            resource_positions.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        values_buffer = self.device.newBufferWithBytes_length_options_(
            resource_values.astype(np.float32).tobytes(),
            resource_values.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create output buffer
        influence_map = np.zeros(grid_size, dtype=np.float32)
        influence_buffer = self.device.newBufferWithBytes_length_options_(
            influence_map.tobytes(),
            influence_map.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create parameter buffers
        grid_size_array = np.array(grid_size, dtype=np.uint32)
        grid_size_buffer = self.device.newBufferWithBytes_length_options_(
            grid_size_array.tobytes(),
            grid_size_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        num_resources_array = np.array([num_resources], dtype=np.uint32)
        num_resources_buffer = self.device.newBufferWithBytes_length_options_(
            num_resources_array.tobytes(),
            num_resources_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        influence_radius_array = np.array([influence_radius], dtype=np.float32)
        radius_buffer = self.device.newBufferWithBytes_length_options_(
            influence_radius_array.tobytes(),
            influence_radius_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Configure pipeline
        compute_encoder.setComputePipelineState_(self.resource_pipeline)
        compute_encoder.setBuffer_offset_atIndex_(positions_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(values_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(influence_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(grid_size_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(num_resources_buffer, 0, 4)
        compute_encoder.setBuffer_offset_atIndex_(radius_buffer, 0, 5)

        # Configure grid and threadgroups
        threads_per_group = min(self.resource_pipeline.maxTotalThreadsPerThreadgroup(), 16)
        threadgroup_size = Metal.MTLSizeMake(threads_per_group, threads_per_group, 1)
        grid_width = (grid_size[0] + threads_per_group - 1) // threads_per_group
        grid_height = (grid_size[1] + threads_per_group - 1) // threads_per_group
        grid_size_metal = Metal.MTLSizeMake(grid_width, grid_height, 1)

        # Dispatch
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            grid_size_metal, threadgroup_size
        )
        compute_encoder.endEncoding()

        # Execute and wait with error handling
        try:
            # Safety check for buffers
            if positions_buffer is None or distances_buffer is None:
                warnings.warn("Invalid Metal buffers, falling back to CPU implementation")
                return self._calculate_distances_cpu(positions)

            # Execute and wait with error catching
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            # Check for errors
            if hasattr(command_buffer, 'error') and command_buffer.error():
                warnings.warn(f"Metal execution error: {command_buffer.error().localizedDescription()}")
                return self._calculate_distances_cpu(positions)
        except Exception as e:
            warnings.warn(f"Exception during Metal execution: {e}")
            return self._calculate_distances_cpu(positions)

        # Copy results back to numpy
        result_ptr = influence_buffer.contents()
        output = np.ctypeslib.as_array(
            (np.ctypeslib.ct.c_float * (grid_size[0] * grid_size[1])).from_address(int(result_ptr)),
            shape=grid_size
        ).copy()

        return output

    def _calculate_resource_influence_cpu(
            self,
            resource_positions: np.ndarray,
            resource_values: np.ndarray,
            grid_size: Tuple[int, int],
            influence_radius: float = 50.0
    ) -> np.ndarray:
        """CPU fallback for resource influence calculation."""
        influence_map = np.zeros(grid_size, dtype=np.float32)
        num_resources = resource_positions.shape[0]

        # For each grid cell
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                pos = np.array([x, y])

                # For each resource
                for i in range(num_resources):
                    resource_pos = resource_positions[i]
                    value = resource_values[i]

                    # Calculate distance
                    diff = pos - resource_pos
                    dist = np.sqrt(np.sum(diff ** 2))

                    # Add influence
                    if dist < influence_radius:
                        factor = 1.0 - (dist / influence_radius)
                        influence_map[x, y] += value * factor ** 2

        return influence_map

    def update_spatial_partition(
            self,
            positions: np.ndarray,
            grid_size: Tuple[int, int],
            cell_size: Tuple[float, float],
            max_entities_per_cell: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update spatial partitioning grid using GPU.

        Args:
            positions: Entity positions with shape (n, 2)
            grid_size: Size of the grid as (width, height)
            cell_size: Size of each cell as (width, height)
            max_entities_per_cell: Maximum entities per cell

        Returns:
            Tuple of (grid_counts, grid_indices)
        """
        if not self.has_metal or not HAS_METAL:
            # Fallback to CPU implementation
            return self._update_spatial_partition_cpu(
                positions, grid_size, cell_size, max_entities_per_cell
            )

        num_entities = positions.shape[0]

        # Create input buffer
        positions_buffer = self.device.newBufferWithBytes_length_options_(
            positions.astype(np.float32).tobytes(),
            positions.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create output buffers
        grid_counts = np.zeros(grid_size[0] * grid_size[1], dtype=np.uint32)
        grid_counts_buffer = self.device.newBufferWithBytes_length_options_(
            grid_counts.tobytes(),
            grid_counts.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        grid_indices = np.zeros(grid_size[0] * grid_size[1] * max_entities_per_cell, dtype=np.uint32)
        grid_indices_buffer = self.device.newBufferWithBytes_length_options_(
            grid_indices.tobytes(),
            grid_indices.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create parameter buffers
        grid_size_array = np.array(grid_size, dtype=np.uint32)
        grid_size_buffer = self.device.newBufferWithBytes_length_options_(
            grid_size_array.tobytes(),
            grid_size_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        cell_size_array = np.array(cell_size, dtype=np.float32)
        cell_size_buffer = self.device.newBufferWithBytes_length_options_(
            cell_size_array.tobytes(),
            cell_size_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        num_entities_array = np.array([num_entities], dtype=np.uint32)
        num_entities_buffer = self.device.newBufferWithBytes_length_options_(
            num_entities_array.tobytes(),
            num_entities_array.nbytes,
            Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Configure pipeline
        compute_encoder.setComputePipelineState_(self.partition_pipeline)
        compute_encoder.setBuffer_offset_atIndex_(positions_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(grid_counts_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(grid_indices_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(grid_size_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(cell_size_buffer, 0, 4)
        compute_encoder.setBuffer_offset_atIndex_(num_entities_buffer, 0, 5)

        # Configure threadgroups
        threads_per_group = min(self.partition_pipeline.maxTotalThreadsPerThreadgroup(), 256)
        threadgroup_size = Metal.MTLSizeMake(threads_per_group, 1, 1)

        # Calculate grid dimensions
        grid_width = (num_entities + threads_per_group - 1) // threads_per_group
        grid_size_metal = Metal.MTLSizeMake(grid_width, 1, 1)

        # Dispatch
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            grid_size_metal, threadgroup_size
        )
        compute_encoder.endEncoding()

        # Execute and wait with error handling
        try:
            # Safety check for buffers
            if positions_buffer is None or distances_buffer is None:
                warnings.warn("Invalid Metal buffers, falling back to CPU implementation")
                return self._calculate_distances_cpu(positions)

            # Execute and wait with error catching
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            # Check for errors
            if hasattr(command_buffer, 'error') and command_buffer.error():
                warnings.warn(f"Metal execution error: {command_buffer.error().localizedDescription()}")
                return self._calculate_distances_cpu(positions)
        except Exception as e:
            warnings.warn(f"Exception during Metal execution: {e}")
            return self._calculate_distances_cpu(positions)

        # Copy results back to numpy
        counts_ptr = grid_counts_buffer.contents()
        indices_ptr = grid_indices_buffer.contents()

        grid_counts_result = np.ctypeslib.as_array(
            (np.ctypeslib.ct.c_uint32 * (grid_size[0] * grid_size[1])).from_address(int(counts_ptr)),
            shape=(grid_size[0] * grid_size[1],)
        ).copy()

        grid_indices_result = np.ctypeslib.as_array(
            (np.ctypeslib.ct.c_uint32 * (grid_size[0] * grid_size[1] * max_entities_per_cell)).from_address(
                int(indices_ptr)),
            shape=(grid_size[0] * grid_size[1], max_entities_per_cell)
        ).copy()

        return grid_counts_result.reshape(grid_size), grid_indices_result.reshape(
            (grid_size[0], grid_size[1], max_entities_per_cell))

    def _update_spatial_partition_cpu(
            self,
            positions: np.ndarray,
            grid_size: Tuple[int, int],
            cell_size: Tuple[float, float],
            max_entities_per_cell: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for spatial partitioning update."""
        num_entities = positions.shape[0]
        grid_counts = np.zeros(grid_size, dtype=np.uint32)
        grid_indices = np.zeros((grid_size[0], grid_size[1], max_entities_per_cell), dtype=np.uint32)

        for entity_id in range(num_entities):
            pos = positions[entity_id]

            # Calculate grid cell
            cell_x = min(int(pos[0] / cell_size[0]), grid_size[0] - 1)
            cell_y = min(int(pos[1] / cell_size[1]), grid_size[1] - 1)

            # Add to grid if there's space
            if grid_counts[cell_x, cell_y] < max_entities_per_cell:
                grid_indices[cell_x, cell_y, grid_counts[cell_x, cell_y]] = entity_id
                grid_counts[cell_x, cell_y] += 1

        return grid_counts, grid_indices


# Singleton instance
metal_compute = MetalCompute()


# Exposed functions
def calculate_distances(positions):
    """Calculate pairwise distances between positions."""
    return metal_compute.calculate_distances(positions)


def calculate_resource_influence(resource_positions, resource_values, grid_size, influence_radius=50.0):
    """Calculate resource influence on environment."""
    return metal_compute.calculate_resource_influence(
        resource_positions, resource_values, grid_size, influence_radius
    )


def update_spatial_partition(positions, grid_size, cell_size, max_entities_per_cell=32):
    """Update spatial partitioning grid."""
    return metal_compute.update_spatial_partition(
        positions, grid_size, cell_size, max_entities_per_cell
    )


def has_metal():
    """Check if Metal is available and initialized."""
    return metal_compute.has_metal