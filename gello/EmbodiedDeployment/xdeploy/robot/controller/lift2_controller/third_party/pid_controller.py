class PIDController:
    def __init__(self, kp, ki, kd, max_i, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_i = max_i
        self.max_output = max_output
        self.previous_error = 0.0
        self.integral = 0.0

    def update(self, current, target, dt):
        # 计算误差
        error = target - current

        # 对小误差应用死区
        error = 0 if abs(error) <= 0.002 else error

        # 计算积分项并防止积分饱和
        self.integral = max(
            min(self.integral + error * dt, self.max_i), -self.max_i
        )

        # 计算微分项
        derivative = (error - self.previous_error) / dt  # 计算微分

        # 计算PID输出
        output = (
            self.kp * error + self.ki * self.integral + self.kd * derivative
        )
        output = max(min(output, self.max_output), -self.max_output)

        # 存储当前误差以供下次使用
        self.previous_error = error

        return output
