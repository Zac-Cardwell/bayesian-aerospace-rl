"""
Rocket Landing Environment
Based on gymnasium's LunarLander but enhanced for aerospace applications.

Features:
- Realistic physics with mass dynamics
- Fuel consumption modeling
- Thrust vectoring control
- Environmental disturbances (wind, sensor noise)
- Engine reliability modeling (thrust dropout)
- Multi-objective rewards (fuel efficiency + safety)

Box2D: 2D physics engine 
Simulates rigid body dynamics, collisions, joints

Box2D.b2: Specific Box2D components:

edgeShape: Creates line segments (for terrain)
fixtureDef: Defines physical properties (density, friction)
polygonShape: Creates polygon collision shapes
revoluteJointDef: Creates rotational joints (for landing legs)
contactListener: Detects when objects touch

b2World: Container for all physics objects
- Handles collision detection, force integration, constraint solving
- Must create bodies and add them to world
- World steps forward in time with world.Step()
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, 
                      revoluteJointDef, contactListener)

# Physical constants
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well -  means 30 pixels = 1 meter
VIEWPORT_W = 600
VIEWPORT_H = 400

# Rocket parameters (SpaceX-inspired)
MAIN_ENGINE_POWER = 40.0  # Main engine thrust
SIDE_ENGINE_POWER = 2.0   # RCS thrusters
INITIAL_MASS = 1.0        # Rocket dry mass
FUEL_MASS = 0.5           # Initial fuel mass
FUEL_CONSUMPTION_MAIN = 0.03  # kg/s for main engine
FUEL_CONSUMPTION_SIDE = 0.005 # kg/s for side engines

# Landing parameters
LANDING_PAD_Y = VIEWPORT_H/SCALE/4
LANDING_PAD_X = VIEWPORT_W/SCALE/2
LEG_DOWN = 18
LEG_AWAY = 20
LEG_SPRING_TORQUE = 40

# Environment challenge parameters
WIND_POWER = 15.0
TURBULENCE_POWER = 2.0

'''
How Box2D collision detection works:

contactListener: Box2D callback interface
- Box2D calls these methods automatically when objects touch
- Must inherit from contactListener to receive callbacks

BeginContact(contact): Called when two objects START touching
- contact: Contains information about the collision
- contact.fixtureA/B: The two objects that collided
- fixture.body: The physical body attached to the fixture
'''
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False




class RocketLandingEnv(gym.Env):
    """
    Rocket landing environment.
    
    Observation Space (13D):
        0: x position (normalized)
        1: y position (normalized)
        2: x velocity
        3: y velocity
        4: angle
        5: angular velocity
        6: left leg contact (binary)
        7: right leg contact (binary)
        8: remaining fuel (normalized)
        9: current mass (normalized)
        10: wind force x
        11: wind force y
        12: time remaining (normalized)
    
    Action Space (3D, continuous):
        0: Main engine throttle [0, 1]
        1: Left RCS throttle [-1, 1]
        2: Right RCS throttle [-1, 1]
    
    Reward Components:
        - Shaping: Progress towards safe landing
        - Fuel efficiency: Penalty for excessive fuel use
        - Safety: Large penalty for crashes, bonus for soft landing
        - Stability: Penalty for high velocities and angles
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }
    
    def __init__(self, 
                 render_mode=None,
                 difficulty='medium',
                 enable_wind=True,
                 enable_sensor_noise=True,
                 enable_thrust_dropout=True,
                 fuel_weight=0.1,
                 max_episode_steps=500):
        """
        Args:
            render_mode: 'human' or 'rgb_array'
            difficulty: 'easy', 'medium', 'hard' (affects initial conditions)
            enable_wind: Enable wind disturbances
            enable_sensor_noise: Add sensor noise to observations
            enable_thrust_dropout: Random engine failures
            fuel_weight: Weight for fuel efficiency in reward [0, 1]
            max_episode_steps: Maximum timesteps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.enable_wind = enable_wind
        self.enable_sensor_noise = enable_sensor_noise
        self.enable_thrust_dropout = enable_thrust_dropout
        self.fuel_weight = fuel_weight
        self.max_episode_steps = max_episode_steps
        
        self.difficulty_settings = {
            'easy': {'initial_random': 0.2, 'wind_scale': 0.5, 'dropout_prob': 0.0},
            'medium': {'initial_random': 0.4, 'wind_scale': 1.0, 'dropout_prob': 0.02},
            'hard': {'initial_random': 0.6, 'wind_scale': 1.5, 'dropout_prob': 0.05}
        }
        
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*13),
            high=np.array([np.inf]*13),
            dtype=np.float64
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float64
        )
        
        self.world = None
        self.lander = None
        self.legs = []
        self.particles = []
        
        self.fuel_remaining = FUEL_MASS
        self.current_mass = INITIAL_MASS + FUEL_MASS
        self.steps = 0
        self.prev_shaping = None
        self.game_over = False
        
        self.wind_force = np.zeros(2)
        self.thrust_dropout = False
    
        self.sensor_noise_std = 0.01 if enable_sensor_noise else 0.0
        
        self.screen = None
        self.clock = None
        self.isopen = True


    def _create_terrain(self):
        """Create landing pad and ground. Divides ground into 11 segments and randomly assigns them heights. 
        5 middle chunks are set to the same height for the landing pad"""
        CHUNKS = 11
        height = np.random.uniform(0, VIEWPORT_H/SCALE/4, size=(CHUNKS+1,))
        chunk_x = [VIEWPORT_W / SCALE / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = LANDING_PAD_Y
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        
        smooth_y = [0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(CHUNKS)]
        
        self.terrain = self.world.CreateStaticBody(
            shapes=[edgeShape(vertices=[(chunk_x[i], smooth_y[i]), (chunk_x[i + 1], smooth_y[i + 1])])
                   for i in range(CHUNKS - 1)]
        )
        
        self.terrain.color1 = (0.5, 0.5, 0.5) # RGB gray
        self.terrain.color2 = (0.3, 0.3, 0.3) # Darker gray


    def _create_rocket(self):
        """Create rocket body with legs. 
        CreateDynamicBody- Makes movable object (vs static terrain)
        """
        settings = self.difficulty_settings[self.difficulty]
        initial_random = settings['initial_random']
        
        # Random initial position
        initial_x = VIEWPORT_W / SCALE / 2 + np.random.uniform(-initial_random, initial_random) * VIEWPORT_W / SCALE
        initial_y = VIEWPORT_H / SCALE * 0.9
        
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=np.random.uniform(-initial_random, initial_random),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in [
                    (-14, +17), (-17, 0), (-17, -10),
                    (+17, -10), (+17, 0), (+14, +17)]]),
                density=1.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0
            )
        )
        
        self.lander.color1 = (0.8, 0.8, 0.8)
        self.lander.color2 = (0.5, 0.5, 0.5)
        
        # Apply random initial velocity
        self.lander.ApplyForceToCenter((
            self.world.gravity[0] * self.lander.mass * np.random.uniform(-initial_random, initial_random),
            self.world.gravity[1] * self.lander.mass * np.random.uniform(-initial_random, initial_random)
        ), True)
        
        # Create landing legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_AWAY / SCALE / 2, LEG_DOWN / SCALE / 2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0010, # What I am (binary: 0001 0000)
                    maskBits=0x001       # What I collide with (binary: 0000 0001)
                )
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.5, 0.5)
            leg.color2 = (0.3, 0.3, 0.3)
            
            '''
            revolute joint
            - Connects two bodies at a point
            - Allows rotation around that point (like a hinge)
            - Can have motor (applies torque) and limits (restricts angle)

            Parameters:
            - bodyA/bodyB: The two connected bodies
            - localAnchorA: Connection point on bodyA (lander) in local coordinates 
                (i * LEG_AWAY / SCALE, 0): At rocket sides, center height
            - localAnchorB: Connection point on bodyB (leg) in local coordinates
                (0, LEG_DOWN / SCALE): At top of leg
            - enableMotor=True: Joint can apply torque
            - maxMotorTorque: Maximum torque (spring strength)
            - motorSpeed: Target rotation speed
                +0.3 * i: Left leg rotates counterclockwise, right clockwise
                Creates "spreading" motion when deployed
            '''
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY / SCALE, 0),
                localAnchorB=(0, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i
            )
            
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
                
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy() # Removes all old Box2D bodiesc
        self.world = Box2D.b2World(gravity=(0, -10))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # Reset state
        self.fuel_remaining = FUEL_MASS
        self.current_mass = INITIAL_MASS + FUEL_MASS
        self.steps = 0
        self.prev_shaping = None
        self.game_over = False
        self.wind_force = np.zeros(2)
        self.thrust_dropout = False
        self._create_terrain()
        self._create_rocket()
        
        return self._get_observation(), {}
    
    def _destroy(self):
        if not self.world:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.terrain)
        self.world.DestroyBody(self.lander)
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.world = None


    def _get_observation(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        x_normalized = (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
        y_normalized = (pos.y - LANDING_PAD_Y) / (VIEWPORT_H/SCALE)

        obs = np.array([
            x_normalized,           # 0: Horizontal position
            y_normalized,           # 1: Vertical position
            vel.x,                  # 2: Horizontal velocity (m/s)
            vel.y,                  # 3: Vertical velocity (m/s)
            self.lander.angle,      # 4: Rotation angle (radians)
            self.lander.angularVelocity,  # 5: Rotation speed (rad/s)
            1.0 if self.legs[0].ground_contact else 0.0,  # 6: Left leg touching
            1.0 if self.legs[1].ground_contact else 0.0,  # 7: Right leg touching
            self.fuel_remaining / FUEL_MASS,  # 8: Fuel % remaining [0,1]
            self.current_mass / (INITIAL_MASS + FUEL_MASS),  # 9: Mass ratio [0.67, 1]
            self.wind_force[0] / WIND_POWER,  # 10: Wind X (normalized)
            self.wind_force[1] / WIND_POWER,  # 11: Wind Y (normalized)
            1.0 - (self.steps / self.max_episode_steps)  # 12: Time remaining [1→0]
        ], dtype=np.float64)

        if self.enable_sensor_noise:
            noise = np.random.normal(0, self.sensor_noise_std, size=obs.shape)
            # Don't add noise to binary signals (leg contacts)
            noise[6:8] = 0  
            obs += noise

        return obs
    

    def _update_disturbances(self):
        """Update wind and other environmental disturbances."""
        if self.enable_wind:
            settings = self.difficulty_settings[self.difficulty]
            wind_scale = settings['wind_scale']
            
            # Persistent wind with turbulence
            self.wind_force[0] = np.random.normal(self.wind_force[0], TURBULENCE_POWER * wind_scale)
            self.wind_force[0] = np.clip(self.wind_force[0], -WIND_POWER * wind_scale, WIND_POWER * wind_scale)
            self.wind_force[1] = np.random.normal(0, TURBULENCE_POWER * wind_scale * 0.5)
            
        # Thrust dropout simulation
        if self.enable_thrust_dropout:
            settings = self.difficulty_settings[self.difficulty]
            dropout_prob = settings['dropout_prob']
            self.thrust_dropout = np.random.random() < dropout_prob


    def _apply_thrust(self, action):
        """Apply thrust forces based on action."""
        main_throttle, left_rcs, right_rcs = action

        '''
        Thrust dropout effect:
        - Reduces thrust to 30% during failure
        - Partial failure more realistic than complete shutdown
        - Rocket must recover from reduced thrust
        '''
        if self.thrust_dropout:
            main_throttle *= 0.3  
            
        fuel_needed = (main_throttle * FUEL_CONSUMPTION_MAIN + 
                      (abs(left_rcs) + abs(right_rcs)) * FUEL_CONSUMPTION_SIDE)
        
        if self.fuel_remaining < fuel_needed:
            # Scale down thrust if insufficient fuel 
            # Makes fuel exhaustion gradual, not instant
            scale = self.fuel_remaining / fuel_needed if fuel_needed > 0 else 0
            main_throttle *= scale
            left_rcs *= scale
            right_rcs *= scale
            fuel_needed *= scale

        # Consume fuel and update mass
        self.fuel_remaining = max(0, self.fuel_remaining - fuel_needed)
        self.current_mass = INITIAL_MASS + self.fuel_remaining

        # Update mass
        # Note: Box2D doesn't support dynamic mass changes easily, 
        # so we approximate by scaling forces
        # F = ma, so a = F/m
        mass_ratio = self.current_mass / (INITIAL_MASS + FUEL_MASS)

        # Main engine
        # When angle = 0 (upright): ox=0, oy=1 (thrust points up)
        # When angle = π/2 (90° right): ox=1, oy=0 (thrust points right)
        # Applying force off-center creates realistic torque - 6 pixels = 0.2m below rocket center
        if main_throttle > 0:
            ox = np.sin(self.lander.angle)
            oy = np.cos(self.lander.angle)
            impulse_pos = (self.lander.position[0] - ox * 6/SCALE, 
                          self.lander.position[1] - oy * 6/SCALE)
            self.lander.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * main_throttle,  # Force in Newtons
                oy * MAIN_ENGINE_POWER * main_throttle),
                impulse_pos,
                True
            )

        # RCS thrusters
        if left_rcs != 0:
            ox = np.sin(self.lander.angle)
            oy = np.cos(self.lander.angle)
            impulse_pos = (self.lander.position[0] - ox * 14/SCALE + oy * 14/SCALE,
                        self.lander.position[1] - oy * 14/SCALE - ox * 14/SCALE)
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * left_rcs / mass_ratio,
                -oy * SIDE_ENGINE_POWER * left_rcs / mass_ratio),
                impulse_pos,
                True
            )
            
        if right_rcs != 0:
            ox = np.sin(self.lander.angle)
            oy = np.cos(self.lander.angle)
            impulse_pos = (self.lander.position[0] - ox * 14/SCALE - oy * 14/SCALE,
                          self.lander.position[1] - oy * 14/SCALE + ox * 14/SCALE)
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * right_rcs / mass_ratio,
                 -oy * SIDE_ENGINE_POWER * right_rcs / mass_ratio),
                impulse_pos,
                True
            )

        
    def _compute_reward(self, action):
        """
        Multi-objective reward function.
        
        Components:
        1. Shaping reward: Progress towards safe landing
        2. Fuel efficiency: Penalty for fuel consumption
        3. Crash penalty: Large negative reward for crashes
        4. Landing bonus: Large positive reward for successful landing
        """
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # Shaping reward (distance and orientation)
        shaping = (
            - 100 * np.sqrt(pos.x - LANDING_PAD_X) ** 2  # Distance to pad
            - 100 * np.sqrt(pos.y - self.helipad_y) ** 2
            - 100 * np.sqrt(vel.x ** 2 + vel.y ** 2)      # Velocity
            - 100 * abs(self.lander.angle)                # Orientation
            + 10 * self.legs[0].ground_contact             # Leg contact
            + 10 * self.legs[1].ground_contact
        )

        '''
        Reward shaping vs terminal rewards:
        - We want reward for improvement, not absolute position
        - shaping - prev_shaping = change in shaping
        - Getting closer to pad → positive reward
        - Moving away → negative reward
        '''
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        else:
            reward = 0
        self.prev_shaping = shaping
        
        # Fuel efficiency penalty
        main_throttle, left_rcs, right_rcs = action
        fuel_penalty = (main_throttle + abs(left_rcs) + abs(right_rcs)) * self.fuel_weight
        reward -= fuel_penalty
        
        # Terminal rewards
        if self.game_over:
            reward = -100  # Crash penalty
        elif not self.lander.awake:
            # Successful landing criteria
            if (self.legs[0].ground_contact and self.legs[1].ground_contact and
                abs(vel.x) < 0.5 and abs(vel.y) < 0.5 and abs(self.lander.angle) < 0.3):
                reward = 100  # Landing bonus
                # Additional fuel efficiency bonus
                reward += 50 * (self.fuel_remaining / FUEL_MASS)
            else:
                reward = -100  # Unstable landing
                
        return reward
    

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update disturbances
        self._update_disturbances()
        
        if self.enable_wind:
            self.lander.ApplyForceToCenter(
                (self.wind_force[0], self.wind_force[1]),
                True
            )
        
        self._apply_thrust(action)
        
        # Physics step
        self.world.Step(1.0 / 50, 6 * 30, 2 * 30)
        
        obs = self._get_observation()
        reward = self._compute_reward(action)
        
        self.steps += 1
        
        terminated = False
        truncated = False
        
        if self.game_over or abs(obs[0]) >= 1.0:
            terminated = True
        elif not self.lander.awake:
            terminated = True
        elif self.steps >= self.max_episode_steps:
            truncated = True
            
        info = {
            'fuel_remaining': self.fuel_remaining,
            'fuel_used': FUEL_MASS - self.fuel_remaining,
            'distance_to_pad': np.sqrt((self.lander.position.x - LANDING_PAD_X)**2 + 
                                      (self.lander.position.y - self.helipad_y)**2),
            'velocity': np.sqrt(self.lander.linearVelocity.x**2 + 
                               self.lander.linearVelocity.y**2),
            'angle': abs(self.lander.angle),
            'crashed': self.game_over,
            'wind_x': self.wind_force[0],
            'wind_y': self.wind_force[1],
            'thrust_dropout': self.thrust_dropout
        }
        
        return obs, reward, terminated, truncated, info
    
    
    def render(self):
        if self.render_mode is None:
            return
            
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            )
            
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            else:
                self.screen = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        
        # Background
        pygame.draw.rect(self.surf, (135, 206, 235), self.surf.get_rect())
        
        # Draw terrain
        for obj in [self.terrain, self.lander] + self.legs:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(v[0] * SCALE, VIEWPORT_H - v[1] * SCALE) for v in path]
                pygame.draw.polygon(self.surf, obj.color1, path)
                
        # Draw wind indicator
        if self.enable_wind:
            wind_start = (50, 50)
            wind_end = (50 + self.wind_force[0] * 2, 50 - self.wind_force[1] * 2)
            pygame.draw.line(self.surf, (255, 0, 0), wind_start, wind_end, 3)
            
        # Draw fuel gauge
        fuel_bar_width = 100
        fuel_bar_height = 20
        fuel_x, fuel_y = 10, VIEWPORT_H - 30
        pygame.draw.rect(self.surf, (100, 100, 100), 
                        (fuel_x, fuel_y, fuel_bar_width, fuel_bar_height))
        fuel_fill = int(fuel_bar_width * (self.fuel_remaining / FUEL_MASS))
        pygame.draw.rect(self.surf, (0, 255, 0), 
                        (fuel_x, fuel_y, fuel_fill, fuel_bar_height))
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
            

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False