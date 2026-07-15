import gymnasium as gym


class AlternateFootWrapper(gym.Wrapper):
    """
    Bestraft den Agenten, wenn die Knie den Boden berühren (Anti-Slide)
    und ermutigt den abwechselnden Gebrauch der Beine.
    """

    def __init__(self, env, slide_penalty=-1.0, hop_penalty=-0.5, ground_threshold=8.5):
        super().__init__(env)
        self.slide_penalty = slide_penalty
        self.hop_penalty = hop_penalty
        self.ground_threshold = ground_threshold

        # Speichert das Bein, das zuletzt einen "echten" Schritt gemacht hat
        self.last_step_leg = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_step_leg = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_data = self.env.unwrapped.last_reaction.data

        left_calf_y = raw_data[(3 * 5) + 1]
        left_thigh_y = raw_data[(6 * 5) + 1]
        right_calf_y = raw_data[(8 * 5) + 1]
        right_thigh_y = raw_data[(11 * 5) + 1]

        # Strafe für Knie-Rutschen (Dein ursprünglicher Code)
        if (left_calf_y > self.ground_threshold or
                right_calf_y > self.ground_threshold or
                left_thigh_y > self.ground_threshold or
                right_thigh_y > self.ground_threshold):
            reward += self.slide_penalty

        # Logik für abwechselnde Füße
        # Wir betrachten ein Bein erst als "getreten", wenn es tiefer als 7.0 ist.
        # (Da 8.5 das Knie-Rutschen ist, ist ~7.5 - 8.0 ein normaler Fuß-Aufsetzer)
        step_threshold = 7.0

        current_leg = None
        if right_calf_y > step_threshold and right_calf_y > left_calf_y:
            current_leg = "Right"
        elif left_calf_y > step_threshold and left_calf_y > right_calf_y:
            current_leg = "Left"

        if current_leg is not None:
            if self.last_step_leg == current_leg:

                reward += self.hop_penalty

            # Update zuletzt benutzte Bein
            self.last_step_leg = current_leg

        return obs, reward, terminated, truncated, info