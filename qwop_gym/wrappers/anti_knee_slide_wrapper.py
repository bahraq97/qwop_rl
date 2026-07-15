import gymnasium as gym


class AntiKneeSlideWrapper(gym.Wrapper):

    """
    Bestraft den Agenten, wenn seine Oberschenkel oder Waden
    (Indikator für Knie) den Boden berühren.
    """

    def __init__(self, env, penalty=-1.0, ground_threshold=8.5):
        # Das Basis-Environment initialisieren
        super().__init__(env)
        self.penalty = penalty
        self.ground_threshold = ground_threshold

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        raw_data = self.env.unwrapped.last_reaction.data

        # 3. Y-Werte der kritischen Körperteile auslesen
        # Basierend auf OBS_PARTS aus dem JS-Wrapper:
        # leftCalf=3, leftThigh=6, rightCalf=8, rightThigh=11
        left_calf_y = raw_data[(3 * 5) + 1]
        left_thigh_y = raw_data[(6 * 5) + 1]
        right_calf_y = raw_data[(8 * 5) + 1]
        right_thigh_y = raw_data[(11 * 5) + 1]

        if (left_calf_y > self.ground_threshold or
                right_calf_y > self.ground_threshold or
                left_thigh_y > self.ground_threshold or
                right_thigh_y > self.ground_threshold):
            reward += self.penalty

        return obs, reward, terminated, truncated, info