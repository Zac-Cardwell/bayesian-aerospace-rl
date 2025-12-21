import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.envs.rocket_landing.rocket_env import RocketLandingEnv

if __name__ == "__main__":
    env_easy = RocketLandingEnv(
        render_mode='human',
        difficulty='easy',
        enable_wind=False,
        enable_sensor_noise=False,
        enable_thrust_dropout=False,
        fuel_weight=0.05
    )
    
    env_hard = RocketLandingEnv(
        render_mode='human',
        difficulty='hard',
        enable_wind=True,
        enable_sensor_noise=True,
        enable_thrust_dropout=True,
        fuel_weight=0.15
    )

    # Test episode
    obs, info = env_easy.reset()
    print(f"Observation space: {env_easy.observation_space}")
    print(f"Action space: {env_easy.action_space}")
    print(f"Initial observation: {obs}")
    
    for _ in range(100):
        action = env_easy.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env_easy.step(action)
        env_easy.render()
        
        if terminated or truncated:
            print(f"Episode finished!")
            print(f"Info: {info}")
            break
            
    env_easy.close()
    