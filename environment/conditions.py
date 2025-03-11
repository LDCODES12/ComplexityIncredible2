"""
Environmental conditions management.
Handles weather, climate, day/night cycles, and seasonal effects.
"""

import os
import sys
import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Try to import noise for terrain generation
try:
    import noise

    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False


class WeatherSystem:
    """
    Manages weather conditions that affect the environment.
    Handles rain, temperature, wind, and other environmental factors.
    """

    def __init__(self, config=None):
        """
        Initialize weather system.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or CONFIG

        # Current weather state
        self.conditions = {
            "temperature": 0.5,  # Normalized 0-1
            "precipitation": 0.0,  # 0=none, 1=heavy
            "wind_speed": 0.0,  # 0=none, 1=storm
            "wind_direction": 0.0,  # Radians
            "cloudiness": 0.0,  # 0=clear, 1=overcast
            "humidity": 0.5,  # 0=dry, 1=humid
            "visibility": 1.0,  # 0=none, 1=perfect
        }

        # Weather event tracking
        self.current_weather_event = None
        self.event_duration = 0
        self.event_intensity = 0.0

        # Historical data
        self.weather_history = []
        self.event_history = []

        # Random seeds for reproducibility
        self.weather_seed = random.randint(0, 1000000)
        self.random_gen = np.random.RandomState(self.weather_seed)

    def update(self, step):
        """
        Update weather conditions for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Updated weather conditions
        """
        # Store previous state for history
        prev_conditions = self.conditions.copy()

        # Check for weather events
        if self.current_weather_event:
            # Update existing event
            self.event_duration -= 1

            if self.event_duration <= 0:
                # End event
                self._end_weather_event()
            else:
                # Continue event
                self._apply_weather_event()
        elif self._should_start_event():
            # Start new event
            self._start_weather_event()
        else:
            # Regular weather progression
            self._update_normal_weather(step)

        # Add to history (every 10 steps)
        if step % 10 == 0:
            self.weather_history.append({
                "step": step,
                "conditions": self.conditions.copy()
            })

        return self.conditions

    def _update_normal_weather(self, step):
        """
        Update weather under normal (non-event) conditions.

        Args:
            step: Current simulation step
        """
        # Apply seasonal and diurnal variations
        self._apply_seasonal_effects(step)
        self._apply_diurnal_effects(step)

        # Add some random variation
        for key in self.conditions:
            if key != "wind_direction":  # Handle direction separately
                # Small random changes
                change = self.random_gen.normal(0, 0.02)
                self.conditions[key] = max(0.0, min(1.0, self.conditions[key] + change))

        # Wind direction changes gradually
        dir_change = self.random_gen.normal(0, 0.1)
        self.conditions["wind_direction"] = (self.conditions["wind_direction"] + dir_change) % (2 * np.pi)

    def _apply_seasonal_effects(self, step):
        """
        Apply seasonal effects to weather conditions.

        Args:
            step: Current simulation step
        """
        # Calculate seasonal progress (365 steps per year)
        seasonal_progress = (step % 365) / 365
        season_value = np.sin(seasonal_progress * 2 * np.pi)

        # Temperature varies with season
        self.conditions["temperature"] = 0.5 + 0.3 * season_value

        # Humidity varies inversely with temperature
        self.conditions["humidity"] = 0.5 - 0.2 * season_value

        # Precipitation more likely in spring/fall
        precip_factor = 1.0 - 2.0 * abs(season_value)
        self.conditions["precipitation"] = max(0.0, min(1.0, 0.2 + 0.3 * precip_factor))

    def _apply_diurnal_effects(self, step):
        """
        Apply day/night cycle effects to weather conditions.

        Args:
            step: Current simulation step
        """
        # Calculate time of day (24 steps per day)
        time_of_day = (step % 24) / 24
        day_night = np.sin(time_of_day * 2 * np.pi)

        # Temperature varies with time of day
        self.conditions["temperature"] += 0.1 * day_night
        self.conditions["temperature"] = max(0.0, min(1.0, self.conditions["temperature"]))

        # Humidity increases at night
        self.conditions["humidity"] += 0.05 * (1 - day_night)
        self.conditions["humidity"] = max(0.0, min(1.0, self.conditions["humidity"]))

        # Visibility decreases at night
        self.conditions["visibility"] = max(0.3, min(1.0, 0.6 + 0.4 * day_night))

    def _should_start_event(self):
        """
        Determine if a weather event should start.

        Returns:
            True if an event should start
        """
        # Base chance for event
        event_chance = 0.005  # 0.5% chance per step

        # Adjust based on conditions
        if self.conditions["precipitation"] > 0.3:
            event_chance *= 1.5  # More likely if already wet

        if self.conditions["temperature"] > 0.7 or self.conditions["temperature"] < 0.3:
            event_chance *= 1.3  # More likely in extreme temperatures

        return random.random() < event_chance

    def _start_weather_event(self):
        """Start a new weather event."""
        # Choose event type
        event_types = [
            "rainstorm", "heatwave", "coldsnap", "windstorm",
            "fog", "snow", "drought", "thunderstorm"
        ]

        # Weight event types by conditions
        weights = np.ones(len(event_types))

        # Adjust weights
        if self.conditions["temperature"] > 0.7:
            # More likely for hot weather events
            weights[event_types.index("heatwave")] *= 3
            weights[event_types.index("drought")] *= 2
            weights[event_types.index("thunderstorm")] *= 1.5
            weights[event_types.index("coldsnap")] *= 0.2
            weights[event_types.index("snow")] *= 0.1
        elif self.conditions["temperature"] < 0.3:
            # More likely for cold weather events
            weights[event_types.index("coldsnap")] *= 3
            weights[event_types.index("snow")] *= 2.5
            weights[event_types.index("heatwave")] *= 0.1
            weights[event_types.index("drought")] *= 0.3

        if self.conditions["humidity"] > 0.7:
            # More likely for wet weather events
            weights[event_types.index("rainstorm")] *= 2
            weights[event_types.index("fog")] *= 1.8
            weights[event_types.index("thunderstorm")] *= 1.5
            weights[event_types.index("drought")] *= 0.2

        # Normalize weights
        weights = weights / np.sum(weights)

        # Choose event
        self.current_weather_event = np.random.choice(event_types, p=weights)

        # Set duration and intensity
        self.event_duration = random.randint(10, 50)  # 10-50 steps
        self.event_intensity = random.uniform(0.5, 1.0)  # 50-100% intensity

        # Log event start
        self.event_history.append({
            "type": self.current_weather_event,
            "start_step": len(self.weather_history) * 10 if self.weather_history else 0,
            "duration": self.event_duration,
            "intensity": self.event_intensity
        })

        # Apply initial event effects
        self._apply_weather_event()

    def _apply_weather_event(self):
        """Apply effects of the current weather event."""
        if self.current_weather_event == "rainstorm":
            self.conditions["precipitation"] = 0.7 + 0.3 * self.event_intensity
            self.conditions["cloudiness"] = 0.8 + 0.2 * self.event_intensity
            self.conditions["visibility"] = 1.0 - 0.5 * self.event_intensity
            self.conditions["humidity"] = 0.8 + 0.2 * self.event_intensity
            self.conditions["wind_speed"] = 0.3 + 0.3 * self.event_intensity

        elif self.current_weather_event == "heatwave":
            self.conditions["temperature"] = 0.8 + 0.2 * self.event_intensity
            self.conditions["humidity"] = max(0.0, self.conditions["humidity"] - 0.3 * self.event_intensity)
            self.conditions["precipitation"] = max(0.0, self.conditions["precipitation"] - 0.2 * self.event_intensity)

        elif self.current_weather_event == "coldsnap":
            self.conditions["temperature"] = 0.2 - 0.15 * self.event_intensity
            self.conditions["wind_speed"] = 0.4 + 0.2 * self.event_intensity

        elif self.current_weather_event == "windstorm":
            self.conditions["wind_speed"] = 0.7 + 0.3 * self.event_intensity
            self.conditions["visibility"] = max(0.3, 1.0 - 0.3 * self.event_intensity)

        elif self.current_weather_event == "fog":
            self.conditions["visibility"] = 0.3 - 0.2 * self.event_intensity
            self.conditions["humidity"] = 0.8 + 0.2 * self.event_intensity
            self.conditions["wind_speed"] = 0.1

        elif self.current_weather_event == "snow":
            self.conditions["temperature"] = 0.2 - 0.1 * self.event_intensity
            self.conditions["precipitation"] = 0.6 + 0.3 * self.event_intensity
            self.conditions["visibility"] = 0.4 + 0.2 * (1.0 - self.event_intensity)
            self.conditions["cloudiness"] = 0.7 + 0.3 * self.event_intensity

        elif self.current_weather_event == "drought":
            self.conditions["precipitation"] = 0.05
            self.conditions["humidity"] = 0.2
            self.conditions["temperature"] = 0.7 + 0.2 * self.event_intensity

        elif self.current_weather_event == "thunderstorm":
            self.conditions["precipitation"] = 0.8 + 0.2 * self.event_intensity
            self.conditions["wind_speed"] = 0.6 + 0.4 * self.event_intensity
            self.conditions["cloudiness"] = 0.9 + 0.1 * self.event_intensity
            self.conditions["visibility"] = 0.3 + 0.2 * (1.0 - self.event_intensity)

    def _end_weather_event(self):
        """End the current weather event."""
        # Update event history
        last_event = self.event_history[-1]
        last_event["end_step"] = len(self.weather_history) * 10 if self.weather_history else 0

        # Clear event
        self.current_weather_event = None
        self.event_duration = 0
        self.event_intensity = 0.0

    def get_season_name(self, step):
        """
        Get the name of the current season.

        Args:
            step: Current simulation step

        Returns:
            Season name as string
        """
        seasonal_progress = (step % 365) / 365
        season_idx = int(seasonal_progress * 4) % 4

        seasons = ["Spring", "Summer", "Fall", "Winter"]
        return seasons[season_idx]

    def get_time_of_day(self, step):
        """
        Get the time of day description.

        Args:
            step: Current simulation step

        Returns:
            Time of day as string
        """
        time_of_day = (step % 24) / 24
        hour = int(time_of_day * 24)

        if 5 <= hour < 8:
            return "Dawn"
        elif 8 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 14:
            return "Noon"
        elif 14 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 21:
            return "Evening"
        elif 21 <= hour < 24:
            return "Night"
        else:  # 0-5
            return "Midnight"

    def get_active_weather_event(self):
        """
        Get information about the current weather event.

        Returns:
            Dictionary with event details or None
        """
        if not self.current_weather_event:
            return None

        return {
            "type": self.current_weather_event,
            "intensity": self.event_intensity,
            "remaining_duration": self.event_duration
        }

    def get_conditions_impact(self):
        """
        Get the impact of current weather on agents and environment.

        Returns:
            Dictionary of impact factors
        """
        impact = {
            "movement_speed": 1.0,  # Multiplier for agent movement speed
            "energy_consumption": 1.0,  # Multiplier for agent energy consumption
            "visibility_range": 1.0,  # Multiplier for agent vision range
            "resource_growth": 1.0,  # Multiplier for resource growth
            "attack_success": 1.0,  # Multiplier for attack success probability
            "social_interaction": 1.0,  # Multiplier for social interaction frequency
        }

        # Apply temperature effects
        if self.conditions["temperature"] < 0.3:
            # Cold weather
            impact["movement_speed"] *= 0.8
            impact["energy_consumption"] *= 1.2
        elif self.conditions["temperature"] > 0.7:
            # Hot weather
            impact["movement_speed"] *= 0.9
            impact["energy_consumption"] *= 1.1

        # Apply precipitation effects
        if self.conditions["precipitation"] > 0.5:
            # Rainy
            impact["movement_speed"] *= 0.8
            impact["visibility_range"] *= 0.7
            impact["resource_growth"] *= 1.2

        # Apply wind effects
        if self.conditions["wind_speed"] > 0.7:
            # Strong wind
            impact["movement_speed"] *= 0.7
            impact["attack_success"] *= 0.8

        # Apply visibility effects
        impact["visibility_range"] *= self.conditions["visibility"]

        # Apply current weather event effects
        if self.current_weather_event:
            if self.current_weather_event == "rainstorm":
                impact["movement_speed"] *= 0.7
                impact["social_interaction"] *= 0.8
                impact["resource_growth"] *= 1.3
            elif self.current_weather_event == "heatwave":
                impact["energy_consumption"] *= 1.3
                impact["resource_growth"] *= 0.6
            elif self.current_weather_event == "coldsnap":
                impact["energy_consumption"] *= 1.4
                impact["movement_speed"] *= 0.7
                impact["resource_growth"] *= 0.5
            elif self.current_weather_event == "windstorm":
                impact["movement_speed"] *= 0.6
                impact["attack_success"] *= 0.7
            elif self.current_weather_event == "fog":
                impact["visibility_range"] *= 0.5
                impact["movement_speed"] *= 0.8
            elif self.current_weather_event == "snow":
                impact["movement_speed"] *= 0.5
                impact["energy_consumption"] *= 1.2
                impact["resource_growth"] *= 0.4
            elif self.current_weather_event == "drought":
                impact["resource_growth"] *= 0.3
            elif self.current_weather_event == "thunderstorm":
                impact["movement_speed"] *= 0.6
                impact["visibility_range"] *= 0.6
                impact["social_interaction"] *= 0.7
                impact["attack_success"] *= 0.7

        return impact


class TerrainGenerator:
    """
    Generates and manages terrain features in the environment.
    Creates height maps, resource distributions, and geographic features.
    """

    def __init__(self, world_size=None, config=None, seed=None):
        """
        Initialize terrain generator.

        Args:
            world_size: (width, height) of the world
            config: Configuration dictionary (optional)
            seed: Random seed for reproducibility (optional)
        """
        self.config = config or CONFIG
        self.world_size = world_size or self.config["world_size"]
        self.seed = seed or random.randint(0, 1000000)

        # Initialize terrain data
        self.height_map = np.zeros(self.world_size, dtype=np.float32)
        self.water_map = np.zeros(self.world_size, dtype=np.float32)
        self.fertility_map = np.zeros(self.world_size, dtype=np.float32)
        self.obstacle_map = np.zeros(self.world_size, dtype=np.bool_)

        # Feature tracking
        self.features = []

        # Set up random generators
        self.random_gen = np.random.RandomState(self.seed)

    def generate_terrain(self):
        """
        Generate complete terrain with all features.

        Returns:
            Dictionary of terrain maps
        """
        # Generate base height map
        self.generate_height_map()

        # Generate water features
        self.generate_water_features()

        # Generate fertility map
        self.generate_fertility_map()

        # Add terrain features
        self.add_terrain_features()

        # Add obstacles
        self.add_obstacles()

        return {
            "height_map": self.height_map,
            "water_map": self.water_map,
            "fertility_map": self.fertility_map,
            "obstacle_map": self.obstacle_map,
            "features": self.features
        }

    def generate_height_map(self, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        Generate terrain height map using noise.

        Args:
            octaves: Number of octaves for noise
            persistence: Persistence parameter for noise
            lacunarity: Lacunarity parameter for noise

        Returns:
            Generated height map
        """
        # Check if noise module is available
        if not HAS_NOISE:
            self._generate_height_map_fallback()
            return self.height_map

        # Generate using Perlin noise
        scale = 100.0
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                nx = x / self.world_size[0] - 0.5
                ny = y / self.world_size[1] - 0.5

                # Add different scales of noise
                self.height_map[x, y] = noise.pnoise2(
                    nx * scale,
                    ny * scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed
                )

        # Normalize to 0-1
        self.height_map = (self.height_map - np.min(self.height_map)) / (
                    np.max(self.height_map) - np.min(self.height_map))

        return self.height_map

    def _generate_height_map_fallback(self):
        """Fallback height map generation without noise module."""
        # Create a simple height map using random walk
        center_x, center_y = self.world_size[0] // 2, self.world_size[1] // 2

        # Start with random mountains
        num_mountains = 10
        for _ in range(num_mountains):
            x = self.random_gen.randint(0, self.world_size[0])
            y = self.random_gen.randint(0, self.world_size[1])
            height = self.random_gen.uniform(0.5, 1.0)
            radius = self.random_gen.randint(20, 100)

            # Create a mountain
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < self.world_size[0] and 0 <= ny < self.world_size[1]:
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist <= radius:
                            factor = 1.0 - (dist / radius)
                            self.height_map[nx, ny] = max(
                                self.height_map[nx, ny],
                                height * factor * factor
                            )

        # Smooth the map
        from scipy.ndimage import gaussian_filter
        self.height_map = gaussian_filter(self.height_map, sigma=3)

        # Normalize to 0-1
        min_val = np.min(self.height_map)
        max_val = np.max(self.height_map)
        if max_val > min_val:
            self.height_map = (self.height_map - min_val) / (max_val - min_val)

        return self.height_map

    def generate_water_features(self, sea_level=0.4, river_count=3):
        """
        Generate water features like oceans, lakes and rivers.

        Args:
            sea_level: Height threshold for water
            river_count: Number of rivers to generate

        Returns:
            Generated water map
        """
        # Ocean and lakes based on height
        self.water_map = np.zeros(self.world_size, dtype=np.float32)

        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if self.height_map[x, y] < sea_level:
                    # Water depth proportional to how far below sea level
                    self.water_map[x, y] = (sea_level - self.height_map[x, y]) / sea_level

        # Generate rivers
        for _ in range(river_count):
            self._generate_river()

        return self.water_map

    def _generate_river(self, min_length=50, max_length=200):
        """
        Generate a river on the map.

        Args:
            min_length: Minimum river length
            max_length: Maximum river length

        Returns:
            List of river points
        """
        # Find a starting point in higher elevation
        for _ in range(100):  # Try 100 times
            x = self.random_gen.randint(0, self.world_size[0] - 1)
            y = self.random_gen.randint(0, self.world_size[1] - 1)

            if self.height_map[x, y] > 0.6:
                break
        else:
            # Couldn't find a good starting point
            return []

        river_points = [(x, y)]
        river_length = self.random_gen.randint(min_length, max_length)

        # Create river by flowing downhill
        for _ in range(river_length):
            x, y = river_points[-1]

            # Look at neighboring points
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.world_size[0] and 0 <= ny < self.world_size[1]:
                    # Check if lower than current
                    if self.height_map[nx, ny] <= self.height_map[x, y]:
                        neighbors.append((nx, ny, self.height_map[nx, ny]))

            if not neighbors:
                break

            # Choose next point based on height (prefer lower points)
            neighbors.sort(key=lambda n: n[2])
            next_x, next_y, _ = neighbors[0]

            # Add to river
            river_points.append((next_x, next_y))

            # Update water map
            self.water_map[next_x, next_y] = 1.0

            # Check if we've reached water
            if self.water_map[next_x, next_y] > 0.5:
                break

        # Widen the river
        for x, y in river_points:
            self.water_map[x, y] = 1.0

            # Add some width
            width = 1 + int(2 * self.random_gen.random())
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < self.world_size[0] and 0 <= ny < self.world_size[1]:
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist <= width:
                            self.water_map[nx, ny] = max(0.0, 1.0 - (dist / width))

        # Add to features list
        self.features.append({
            "type": "river",
            "points": river_points,
            "length": len(river_points)
        })

        return river_points

    def generate_fertility_map(self):
        """
        Generate a fertility map for plant growth.

        Returns:
            Generated fertility map
        """
        # Initialize fertility map
        self.fertility_map = np.zeros(self.world_size, dtype=np.float32)

        # Base fertility on height (middle heights are most fertile)
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                # Most fertile at mid-heights, less at extremes
                h = self.height_map[x, y]
                if self.water_map[x, y] > 0.5:
                    # Water is not fertile
                    self.fertility_map[x, y] = 0.0
                else:
                    # Peak fertility around 0.6 height
                    h_factor = 1.0 - 2.0 * abs(h - 0.6)
                    self.fertility_map[x, y] = max(0.0, h_factor)

        # Areas near water are more fertile
        from scipy.ndimage import gaussian_filter
        water_influence = gaussian_filter(self.water_map, sigma=3)

        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if self.water_map[x, y] < 0.5:  # Not in water
                    water_factor = water_influence[x, y]
                    self.fertility_map[x, y] += 0.3 * water_factor

        # Add some noise for variation
        if HAS_NOISE:
            scale = 50.0
            for y in range(self.world_size[1]):
                for x in range(self.world_size[0]):
                    nx = x / self.world_size[0] - 0.5
                    ny = y / self.world_size[1] - 0.5

                    noise_val = noise.pnoise2(
                        nx * scale,
                        ny * scale,
                        octaves=3,
                        persistence=0.5,
                        lacunarity=2.0,
                        repeatx=1024,
                        repeaty=1024,
                        base=self.seed + 1
                    )

                    # Normalize noise to 0-1
                    noise_val = (noise_val + 1) / 2

                    # Add to fertility with reduced impact
                    self.fertility_map[x, y] += 0.2 * noise_val
        else:
            # Simple random noise
            noise_array = self.random_gen.rand(*self.world_size) * 0.2
            self.fertility_map += noise_array

        # Normalize to 0-1
        self.fertility_map = np.clip(self.fertility_map, 0, 1)

        return self.fertility_map

    def add_terrain_features(self, num_features=10):
        """
        Add special terrain features like forests, swamps, etc.

        Args:
            num_features: Number of features to add

        Returns:
            List of added features
        """
        feature_types = ["forest", "swamp", "hills", "desert", "plains"]

        for _ in range(num_features):
            # Choose feature type
            feature_type = self.random_gen.choice(feature_types)

            # Find suitable location based on terrain type
            for attempt in range(100):
                x = self.random_gen.randint(0, self.world_size[0] - 1)
                y = self.random_gen.randint(0, self.world_size[1] - 1)

                if self._is_suitable_for_feature(x, y, feature_type):
                    break
            else:
                continue  # Skip if no suitable location found

            # Generate feature
            radius = self.random_gen.randint(20, 80)
            self._create_terrain_feature(x, y, radius, feature_type)

        return self.features

    def _is_suitable_for_feature(self, x, y, feature_type):
        """
        Check if a location is suitable for a feature.

        Args:
            x, y: Coordinates to check
            feature_type: Type of feature

        Returns:
            True if suitable
        """
        # Check basic conditions
        if self.water_map[x, y] > 0.5:
            return False  # Not in water

        # Check specific conditions
        if feature_type == "forest":
            return self.fertility_map[x, y] > 0.6 and 0.4 < self.height_map[x, y] < 0.8
        elif feature_type == "swamp":
            return self.fertility_map[x, y] > 0.4 and self.height_map[x, y] < 0.5
        elif feature_type == "hills":
            return 0.6 < self.height_map[x, y] < 0.8
        elif feature_type == "desert":
            return self.fertility_map[x, y] < 0.3 and self.height_map[x, y] > 0.4
        elif feature_type == "plains":
            return self.fertility_map[x, y] > 0.4 and 0.4 < self.height_map[x, y] < 0.7

        return False

    def _create_terrain_feature(self, center_x, center_y, radius, feature_type):
        """
        Create a terrain feature centered at the given location.

        Args:
            center_x, center_y: Center coordinates
            radius: Feature radius
            feature_type: Type of feature

        Returns:
            Feature data
        """
        feature_points = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = center_x + dx, center_y + dy

                if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist <= radius:
                        feature_points.append((x, y))

                        # Apply feature effects
                        factor = 1.0 - (dist / radius)

                        if feature_type == "forest":
                            self.fertility_map[x, y] = max(self.fertility_map[x, y], 0.7 * factor + 0.3)
                        elif feature_type == "swamp":
                            self.fertility_map[x, y] = max(self.fertility_map[x, y], 0.5 * factor + 0.2)
                            # Partial water
                            self.water_map[x, y] = max(self.water_map[x, y], 0.3 * factor)
                        elif feature_type == "hills":
                            self.height_map[x, y] = min(1.0, self.height_map[x, y] + 0.2 * factor)
                        elif feature_type == "desert":
                            self.fertility_map[x, y] = min(self.fertility_map[x, y], 0.3 * (1.0 - factor))
                        elif feature_type == "plains":
                            self.fertility_map[x, y] = max(self.fertility_map[x, y], 0.6 * factor + 0.2)
                            # Flatten height
                            target_height = 0.5
                            self.height_map[x, y] = self.height_map[x, y] * (1.0 - factor) + target_height * factor

        # Add to features list
        feature = {
            "type": feature_type,
            "center": (center_x, center_y),
            "radius": radius,
            "points": feature_points
        }

        self.features.append(feature)
        return feature

    def add_obstacles(self, obstacle_ratio=0.05):
        """
        Add impassable obstacles to the map.

        Args:
            obstacle_ratio: Ratio of map area to cover with obstacles

        Returns:
            Updated obstacle map
        """
        # Reset obstacle map
        self.obstacle_map = np.zeros(self.world_size, dtype=np.bool_)

        # Mountains (high terrain) are obstacles
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if self.height_map[x, y] > 0.8:
                    self.obstacle_map[x, y] = True

        # Deep water is an obstacle
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if self.water_map[x, y] > 0.7:
                    self.obstacle_map[x, y] = True

        # Add some random obstacles
        target_obstacles = int(obstacle_ratio * self.world_size[0] * self.world_size[1])
        current_obstacles = np.sum(self.obstacle_map)

        if current_obstacles < target_obstacles:
            to_add = target_obstacles - current_obstacles

            # Add obstacle clusters
            while to_add > 0:
                x = self.random_gen.randint(0, self.world_size[0] - 1)
                y = self.random_gen.randint(0, self.world_size[1] - 1)

                # Don't place obstacles in water
                if self.water_map[x, y] > 0.5:
                    continue

                # Create an obstacle cluster
                radius = self.random_gen.randint(5, 15)
                added = 0

                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < self.world_size[0] and 0 <= ny < self.world_size[1]:
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist <= radius and self.random_gen.random() < 0.7:
                                if not self.obstacle_map[nx, ny]:
                                    self.obstacle_map[nx, ny] = True
                                    added += 1
                                    to_add -= 1

                                    if to_add <= 0:
                                        break

                    if to_add <= 0:
                        break

                # Add to features list if significant
                if added > 10:
                    self.features.append({
                        "type": "obstacle",
                        "center": (x, y),
                        "radius": radius,
                        "count": added
                    })

        return self.obstacle_map

    def get_movement_cost_map(self):
        """
        Generate a map of movement costs based on terrain.

        Returns:
            Movement cost map (higher = harder to traverse)
        """
        movement_cost = np.ones(self.world_size, dtype=np.float32)

        # Obstacles are impassable
        movement_cost[self.obstacle_map] = np.inf

        # Water increases cost
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if self.water_map[x, y] > 0.0:
                    # Shallow water is traversable but costly
                    if self.water_map[x, y] < 0.7:
                        movement_cost[x, y] = 1.0 + 2.0 * self.water_map[x, y]
                    else:
                        movement_cost[x, y] = np.inf  # Deep water impassable

        # Height increases cost
        for y in range(self.world_size[1]):
            for x in range(self.world_size[0]):
                if movement_cost[x, y] < np.inf:  # Not already impassable
                    # Higher terrain is harder to traverse
                    if self.height_map[x, y] > 0.6:
                        movement_cost[x, y] += 2.0 * (self.height_map[x, y] - 0.6)

        # Apply feature effects
        for feature in self.features:
            if feature["type"] == "forest":
                for x, y in feature["points"]:
                    if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
                        if movement_cost[x, y] < np.inf:  # Not already impassable
                            movement_cost[x, y] *= 1.3  # Forests slow movement

            elif feature["type"] == "swamp":
                for x, y in feature["points"]:
                    if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
                        if movement_cost[x, y] < np.inf:  # Not already impassable
                            movement_cost[x, y] *= 1.5  # Swamps slow movement significantly

        return movement_cost

    def find_starting_positions(self, num_positions, min_distance=50):
        """
        Find good starting positions for agents.

        Args:
            num_positions: Number of positions to find
            min_distance: Minimum distance between positions

        Returns:
            List of (x, y) positions
        """
        positions = []
        attempts = 0
        max_attempts = 1000

        while len(positions) < num_positions and attempts < max_attempts:
            attempts += 1

            # Generate a candidate position
            x = self.random_gen.randint(0, self.world_size[0] - 1)
            y = self.random_gen.randint(0, self.world_size[1] - 1)

            # Check if position is valid (not obstacle or water)
            if self.obstacle_map[x, y] or self.water_map[x, y] > 0.5:
                continue

            # Check minimum distance to existing positions
            too_close = False
            for px, py in positions:
                if np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_distance:
                    too_close = True
                    break

            if not too_close:
                positions.append((x, y))

        return positions


class Environment:
    """
    Environment combining weather and terrain systems.
    Provides overall environmental conditions.
    """

    def __init__(self, config=None):
        """
        Initialize environment.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or CONFIG

        # Initialize subsystems
        self.weather = WeatherSystem(self.config)
        self.terrain = TerrainGenerator(self.config["world_size"], self.config)

        # Generate terrain
        self.terrain_data = self.terrain.generate_terrain()

        # Current conditions
        self.conditions = {
            "temperature": 0.5,  # Normalized 0-1
            "season": 0,  # 0=spring, 1=summer, 2=fall, 3=winter
            "day_night": 0.5,  # 0=night, 1=day
            "weather": "clear",  # Weather description
            "disaster_risk": 0.0  # Probability of disaster
        }

    def update(self, step):
        """
        Update environment for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Updated environmental conditions
        """
        # Update weather
        weather_conditions = self.weather.update(step)

        # Update environmental conditions
        self._update_conditions(step, weather_conditions)

        return self.conditions

    def _update_conditions(self, step, weather_conditions):
        """
        Update overall environmental conditions.

        Args:
            step: Current simulation step
            weather_conditions: Weather conditions from weather system
        """
        # Day-night cycle (period of 24 steps)
        self.conditions["day_night"] = (np.sin(step * np.pi / 12) + 1) / 2

        # Seasonal cycle (period of 365 steps)
        season_progress = (step % 365) / 365
        self.conditions["season"] = int(season_progress * 4) % 4

        # Temperature varies with season and day-night
        season_temp = np.sin(season_progress * 2 * np.pi)
        day_temp = (self.conditions["day_night"] - 0.5) * 0.3
        base_temp = 0.5 + 0.3 * season_temp + day_temp

        # Apply weather temperature
        weather_impact = weather_conditions["temperature"] - 0.5
        self.conditions["temperature"] = max(0.0, min(1.0, base_temp + weather_impact))

        # Weather description
        if self.weather.current_weather_event:
            self.conditions["weather"] = self.weather.current_weather_event
        elif weather_conditions["precipitation"] > 0.6:
            self.conditions["weather"] = "rainy"
        elif weather_conditions["precipitation"] > 0.3:
            self.conditions["weather"] = "drizzle"
        elif weather_conditions["cloudiness"] > 0.7:
            self.conditions["weather"] = "cloudy"
        elif weather_conditions["cloudiness"] > 0.3:
            self.conditions["weather"] = "partly_cloudy"
        else:
            self.conditions["weather"] = "clear"

        # Disaster risk
        if self.weather.current_weather_event in ["thunderstorm", "windstorm",
                                                  "rainstorm"] and self.weather.event_intensity > 0.8:
            self.conditions["disaster_risk"] = 0.2 * self.weather.event_intensity
        else:
            # Random disasters
            if random.random() < 0.001:  # 0.1% chance per step
                self.conditions["disaster_risk"] = random.uniform(0.5, 1.0)
            else:
                self.conditions["disaster_risk"] *= 0.95  # Decay risk over time

    def get_movement_cost(self, position):
        """
        Get the movement cost at a position.

        Args:
            position: (x, y) position

        Returns:
            Movement cost multiplier (1.0 = normal)
        """
        x, y = int(position[0]), int(position[1])

        # Check bounds
        if not (0 <= x < self.config["world_size"][0] and 0 <= y < self.config["world_size"][1]):
            return float('inf')

        # Get base movement cost from terrain
        movement_cost_map = self.terrain.get_movement_cost_map()
        base_cost = movement_cost_map[x, y]

        if base_cost == float('inf'):
            return float('inf')

        # Apply weather effects
        weather_impact = self.weather.get_conditions_impact()
        weather_movement_multiplier = weather_impact["movement_speed"]

        return base_cost / weather_movement_multiplier

    def is_position_valid(self, position):
        """
        Check if a position is valid for an agent to occupy.

        Args:
            position: (x, y) position

        Returns:
            True if position is valid
        """
        x, y = int(position[0]), int(position[1])

        # Check bounds
        if not (0 <= x < self.config["world_size"][0] and 0 <= y < self.config["world_size"][1]):
            return False

        # Check for obstacles
        if self.terrain.obstacle_map[x, y]:
            return False

        # Check for deep water
        if self.terrain.water_map[x, y] > 0.7:
            return False

        return True

    def get_fertile_positions(self, count=1, min_fertility=0.7):
        """
        Get positions with high fertility.

        Args:
            count: Number of positions to return
            min_fertility: Minimum fertility level

        Returns:
            List of (x, y) positions
        """
        positions = []

        # Find positions with fertility above threshold
        candidates = []
        for y in range(self.config["world_size"][1]):
            for x in range(self.config["world_size"][0]):
                if (self.terrain.fertility_map[x, y] >= min_fertility and
                        not self.terrain.obstacle_map[x, y] and
                        self.terrain.water_map[x, y] < 0.5):
                    candidates.append((x, y))

        # Sample randomly
        if candidates:
            positions = random.sample(candidates, min(count, len(candidates)))

        return positions

    def get_environment_effects(self):
        """
        Get the overall effects of current environmental conditions.

        Returns:
            Dictionary of effect multipliers
        """
        # Combine weather and environmental effects
        weather_impact = self.weather.get_conditions_impact()

        effects = weather_impact.copy()

        # Add seasonal effects
        if self.conditions["season"] == 0:  # Spring
            effects["resource_growth"] *= 1.2
        elif self.conditions["season"] == 1:  # Summer
            effects["energy_consumption"] *= 1.1
        elif self.conditions["season"] == 2:  # Fall
            effects["resource_growth"] *= 0.9
        elif self.conditions["season"] == 3:  # Winter
            effects["energy_consumption"] *= 1.2
            effects["resource_growth"] *= 0.7

        # Add time-of-day effects
        if self.conditions["day_night"] < 0.3:  # Night
            effects["visibility_range"] *= 0.7
            effects["movement_speed"] *= 0.9
            effects["attack_success"] *= 0.8

        return effects

    def get_rich_resource_area(self, resource_type=None):
        """
        Find an area with high resource potential.

        Args:
            resource_type: Type of resource (optional)

        Returns:
            (x, y) position of rich area
        """
        # Use terrain fertility as a proxy for resource potential
        if resource_type == "food":
            # For food, find high fertility areas
            x, y = np.unravel_index(np.argmax(self.terrain.fertility_map), self.terrain.fertility_map.shape)
            return (int(x), int(y))
        elif resource_type == "knowledge":
            # For knowledge, prefer varied terrain near water
            # This is just a heuristic
            water_proximity = gaussian_filter(self.terrain.water_map, sigma=5)
            height_variation = np.zeros_like(self.terrain.height_map)

            # Calculate height variation in local area
            for y in range(1, self.terrain.height_map.shape[1] - 1):
                for x in range(1, self.terrain.height_map.shape[0] - 1):
                    neighbors = [
                        self.terrain.height_map[x - 1, y],
                        self.terrain.height_map[x + 1, y],
                        self.terrain.height_map[x, y - 1],
                        self.terrain.height_map[x, y + 1]
                    ]
                    height_variation[x, y] = np.std(neighbors)

            # Combine factors
            knowledge_potential = height_variation * water_proximity
            x, y = np.unravel_index(np.argmax(knowledge_potential), knowledge_potential.shape)
            return (int(x), int(y))
        else:
            # Generic resource area based on fertility
            x, y = np.unravel_index(np.argmax(self.terrain.fertility_map), self.terrain.fertility_map.shape)
            return (int(x), int(y))
    def get_nearby_resources(self, position, radius):
        """
        Get resources near a position.
        This is a compatibility method that delegates to the resources component.

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of resources within radius
        """
        # This is a compatibility method - should delegate to ResourceManager
        # In this implementation we need to return an empty list as we don't have
        # direct access to the ResourceManager from the Environment class
        return []
