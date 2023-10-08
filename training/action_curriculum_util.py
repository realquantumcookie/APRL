from typing import Optional, Tuple

def get_action_curriculum_planner_linear(
    action_curriculum_steps: int,
    action_curriculum_start: float,
    action_curriculum_end: float,
    start_step: int = 0,
):
    def planned_action_low_high(step: int) -> Optional[Tuple[float, float]]:
        if action_curriculum_steps <= 0:
            return None
        if step - start_step < action_curriculum_steps:
            alpha = (step - start_step) / action_curriculum_steps
            current_action = (1 - alpha) * action_curriculum_start + alpha * action_curriculum_end
            return -current_action, current_action
        else:
            current_action = action_curriculum_end
            return -current_action, current_action
    return planned_action_low_high

def get_action_curriculum_planner_quadratic(
    action_curriculum_steps: int,
    action_curriculum_start: float,
    action_curriculum_end: float,
    start_step: int = 0,
):
    def planned_action_low_high(step: int) -> Optional[Tuple[float, float]]:
        if action_curriculum_steps <= 0:
            return None
        if step - start_step < action_curriculum_steps:
            alpha = (step - start_step) / action_curriculum_steps
            alpha_2 = alpha * alpha
            current_action = (1 - alpha_2) * action_curriculum_start + alpha_2 * action_curriculum_end
            return -current_action, current_action
        else:
            current_action = action_curriculum_end
            return -current_action, current_action
    return planned_action_low_high
    