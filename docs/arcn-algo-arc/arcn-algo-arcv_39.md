# Verlet 积分

> [`www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html`](https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html)

Verlet 积分实际上是解决任何物体运动运动学方程的解决方案，

其中是位置，是速度，是加速度，是经常被遗忘的加速度项，是时间。这个方程是几乎每个牛顿物理求解器的核心方程，并引发了一类被称为*力积分器*的算法。第一个与之合作的力积分器是*Verlet 积分*。

所以，假设我们想要求解在中的下一个时间步。为了近似（实际上是在关于进行泰勒级数展开），这可能看起来像这样：

这意味着如果我们需要找到下一个，我们需要当前的，，等。然而，由于很少有人计算加速度项，我们的误差通常为。话虽如此，如果我们玩点小技巧，我们可以用更少的知识和更高的精度来计算！假设我们想要计算*上一个*时间步的。再次，为了近似，这可能看起来像这样：

现在，我们有两个方程要解，分别对应 x 轴上的两个不同时间步，其中一个我们已经有了。如果我们把这两个方程加起来并解出，我们发现

因此，这意味着我们可以通过知道当前的，之前的时间步的，以及加速度来简单地找到我们的下一个！不需要速度！此外，这还将误差降低到，这真是太好了！以下是代码中的样子：

```
function verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0

    while (pos > 0)
        time += dt
        temp_pos = pos
        pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    end

    return time
end 
```

```
double verlet(double pos, double acc, double dt) {

  double prev_pos = pos;
  double time = 0;

  while (pos > 0) {
    time += dt;
    double next_pos = pos * 2 - prev_pos + acc * dt * dt;
    prev_pos = pos;
    pos = next_pos;
  }

  return time;
} 
```

```
void verlet(double *time, double pos, double acc, double dt) {
    double prev_pos, temp_pos;
    prev_pos = pos;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        temp_pos = pos;
        pos = pos * 2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;
    }
} 
```

```
static double verlet(double pos, double acc, double dt) {

  // Note that we are using a temp variable for the previous position
  double prev_pos, temp_pos, time;
  prev_pos = pos;
  time = 0;

  while (pos > 0) {
        time += dt;
        temp_pos = pos;
        pos = pos*2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;
    }

    return time;
} 
```

```
def verlet(pos, acc, dt):
    prev_pos = pos
    time = 0

    while pos > 0:
        time += dt
        next_pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos, pos = pos, next_pos

    return time 
```

```
type Method = Model -> Time -> Particle -> Particle -> Particle

verlet :: Method
verlet acc dt (xOld, _, _, _) (x, _, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = 0
    a' = acc (x', v', a, t + dt) 
```

```
function verlet(pos, acc, dt) {
  let prevPos = pos;
  let time = 0;
  let tempPos;

  while (pos > 0) {
    time += dt;
    tempPos = pos;
    pos = pos * 2 - prevPos + acc * dt * dt;
    prevPos = tempPos;
  }

  return time;
} 
```

```
fn verlet(mut pos: f64, acc: f64, dt: f64) -> f64 {
    let mut prev_pos = pos;
    let mut time = 0.0;

    while pos > 0.0 {
        time += dt;
        let temp_pos = pos;
        pos = pos * 2.0 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;
    }

    time
} 
```

```
func verlet(pos: Double, acc: Double, dt: Double) -> Double {
    var pos = pos
    var temp_pos, time: Double
    var prev_pos = pos
    time = 0.0

    while (pos > 0) {
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    }

    return time
} 
```

```
SUBROUTINE verlet(pos, acc, dt, time) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time
    REAL(8)                :: prev_pos, next_pos

    prev_pos = pos
    time     = 0d0

    DO
        IF (pos > 0d0) THEN
            time     = time + dt
            next_pos = pos * 2d0 - prev_pos + acc * dt ** 2
            prev_pos = pos
            pos      = next_pos
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE verlet 
```

```
def verlet(pos, acc, dt)

    prev_pos = pos
    time = 0
    while pos > 0 do
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    end

   return time

end 
```

```
func verlet(pos, acc, dt float64) (time float64) {
    prevPos := pos
    time = 0

    for pos > 0 {
        time += dt
        nextPos := pos*2 - prevPos + acc*dt*dt
        prevPos, pos = pos, nextPos
    }

    return
} 
```

```
# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    verlet_return
  movsd  xmm6, xmm1                  # xmm6 = acc * dt * dt
  mulsd  xmm6, xmm2
  mulsd  xmm6, xmm2
  movsd  xmm5, xmm0                  # Holds previous position
verlet_loop:
  addsd  xmm3, xmm2                  # Adding dt to time
  movsd  xmm4, xmm0                  # Hold old value of posistion
  addsd  xmm0, xmm0                  # Calculating new position
  subsd  xmm0, xmm5
  addsd  xmm0, xmm6
  movsd  xmm5, xmm4
  comisd xmm0, xmm7                  # Check if position is greater then 0.0
  ja     verlet_loop
verlet_return:
  movsd  xmm0, xmm3                  # Saving time value
  ret 
```

```
fun verlet(_pos: Double, acc: Double, dt: Double): Double {
    var pos = _pos  // Since function parameter are val and can't be modified
    var prevPos = pos
    var time = 0.0

    while (pos > 0) {
        time += dt
        val nextPos = pos * 2 - prevPos + acc * dt * dt
        prevPos = pos
        pos = nextPos
    }
    return time
} 
```

```
func verlet(pos_in, acc, dt: float): float =
  var
    pos: float = pos_in
    prevPos: float = pos
    time: float = 0.0
    tempPos: float

  while pos > 0.0:
    time += dt
    tempPos = pos
    pos = pos * 2 - prevPos + acc * dt * dt
    prevPos = tempPos

  time 
```

```
(defun verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using Verlet integration."
  (loop
    with prev-pos = pos
    for time = 0 then (incf time dt)
    while (> pos 0)
    ;; The starting speed is assumed to be zero.
    do (psetf
         pos (+ (* pos 2) (- prev-pos) (* acc dt dt))
         prev-pos pos)
    finally (return time))) 
```

显然，这提出了一个问题；如果我们想要计算一个需要速度的项，比如动能，怎么办？在这种情况下，我们当然不能去掉速度！我们可以通过使用 Stormer-Verlet 方法找到速度到精度。我们上面有关于和的方程，所以让我们从这里开始。如果我们从后者减去前者，我们得到以下方程：

当我们求解时，我们得到

注意，分母中的 2 是有意义的，因为我们正在跨越两个时间步。这本质上是在解。此外，我们可以这样计算下一个时间步的速度

然而，这个误差为，这相当糟糕，但在紧急情况下可以完成任务。以下是代码中的样子：

```
function stormer_verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0
    vel = 0.0

    while (pos > 0.0)
        time += dt
        temp_pos = pos
        pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        # Because acceleration is constant, velocity is straightforward
        vel += acc * dt
    end

    return time, vel
end 
```

```
timestep stormer_verlet(double pos, double acc, double dt) {

  double prev_pos = pos;
  double time = 0;
  double vel = 0;
  while (pos > 0) {
    time += dt;
    double next_pos = pos * 2 - prev_pos + acc * dt * dt;
    prev_pos = pos;
    pos = next_pos;

    // The acceleration is constant, so the velocity is
    // straightforward
    vel += acc * dt;
  }

  return timestep { time, vel };
} 
```

```
void stormer_verlet(double *time, double *vel,
                    double pos, double acc, double dt) {
    double prev_pos, temp_pos;
    prev_pos = pos;
    *vel = 0.0;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        temp_pos = pos;
        pos = pos * 2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;

        *vel += acc * dt;
    }
} 
```

```
static VerletValues stormer_verlet(double pos, double acc, double dt) {

    // Note that we are using a temp variable for the previous position
    double prev_pos, temp_pos, time, vel;
    prev_pos = pos;
    vel = 0;
    time = 0;
    while (pos > 0) {
        time += dt;
        temp_pos = pos;
        pos = pos*2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;

        // The acceleration is constant, so the velocity is straightforward
         vel += acc*dt;
    }

   return new VerletValues(time, vel);
} 
```

```
def stormer_verlet(pos, acc, dt):
    prev_pos = pos
    time = 0
    vel = 0

    while pos > 0:
        time += dt
        next_pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos, pos = pos, next_pos
        vel += acc * dt

    return time, vel 
```

```
stormerVerlet :: Method
stormerVerlet acc dt (xOld, _, _, _) (x, _, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = (x' - x) / dt
    a' = acc (x', v', a, t + dt) 
```

```
function stormerVerlet(pos, acc, dt) {
  let prevPos = pos;
  let time = 0;
  let vel = 0;
  let tempPos;

  while (pos > 0) {
    time += dt;
    tempPos = pos;
    pos = pos * 2 - prevPos + acc * dt * dt;
    prevPos = tempPos;

    vel += acc * dt;
  }

  return { time, vel };
} 
```

```
fn stormer_verlet(mut pos: f64, acc: f64, dt: f64) -> (f64, f64) {
    let mut prev_pos = pos;
    let mut time = 0.0;
    let mut vel = 0.0;

    while pos > 0.0 {
        time += dt;
        let temp_pos = pos;
        pos = pos * 2.0 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;

        // Because acceleration is constant, velocity is
        // straightforward
        vel += acc * dt;
    }

    (time, vel)
} 
```

```
func stormerVerlet(pos: Double, acc: Double, dt: Double) -> (time: Double, vel: Double) {
    var pos = pos
    var temp_pos, time, vel: Double
    var prev_pos = pos
    vel = 0
    time = 0

    while (pos > 0) {
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        vel += acc*dt
    }

    return (time:time, vel:vel)
} 
```

```
SUBROUTINE stormer_verlet(pos, acc, dt, time, vel) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel
    REAL(8)                :: prev_pos, next_pos

    prev_pos = pos 
    time     = 0d0
    vel      = 0d0

    DO
        IF (pos > 0d0) THEN
            time     = time + dt
            next_pos = pos * 2 - prev_pos + acc * dt ** 2
            prev_pos = pos
            pos      = next_pos
            vel      = vel + acc * dt
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE stormer_verlet 
```

```
def stormer_verlet(pos, acc, dt)

    prev_pos = pos
    vel = 0
    time = 0
    while pos > 0 do
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        vel += acc*dt
    end

   return time, vel

end 
```

```
func stormerVerlet(pos, acc, dt float64) (time, vel float64) {
    prevPos := pos
    time, vel = 0, 0

    for pos > 0 {
        time += dt
        vel += acc * dt
        nextPos := pos*2 - prevPos + acc*dt*dt
        prevPos, pos = pos, nextPos
    }

    return
} 
```

```
# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
# RET xmm1 - velocity
stormer_verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    stormer_verlet_return
  movsd  xmm6, xmm1                  # xmm6 = acc * dt * dt
  mulsd  xmm6, xmm2
  mulsd  xmm6, xmm2
  movsd  xmm5, xmm0                  # Holds previous position
stormer_verlet_loop:
  addsd  xmm3, xmm2                  # Adding dt to time
  movsd  xmm4, xmm0                  # Hold old value of posistion
  addsd  xmm0, xmm0                  # Calculating new position
  subsd  xmm0, xmm5
  addsd  xmm0, xmm6
  movsd  xmm5, xmm4
  comisd xmm0, xmm7                  # Check if position is greater then 0.0
  ja     stormer_verlet_loop
stormer_verlet_return:
  movsd  xmm0, xmm3                  # Saving time and velocity
  mulsd  xmm3, xmm1
  movsd  xmm1, xmm3
  ret 
```

```
fun stormerVerlet(_pos: Double, acc: Double, dt: Double): VerletValues {
    var pos = _pos
    var prevPos = pos
    var time = 0.0
    var vel = 0.0
    while (pos > 0) {
        time += dt
        val nextPos = pos * 2 - prevPos + acc * dt * dt
        prevPos = pos
        pos = nextPos
        vel += acc * dt
    }
    return VerletValues(time, vel)
} 
```

```
func stormerVerlet(pos_in, acc, dt: float): (float, float) =
  var
    pos: float = pos_in
    prevPos: float = pos
    time: float = 0.0
    vel: float = 0.0
    tempPos: float

  while pos > 0.0:
    time += dt
    tempPos = pos
    pos = pos * 2 - prevPos + acc * dt * dt
    prevPos = tempPos

    vel += acc * dt

  (time, vel) 
```

```
(defun stormer-verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using the Stormer-Verlet method."
  (loop
    with prev-pos = pos
    for time = 0 then (incf time dt)
    for vel = 0 then (incf vel (* acc dt))
    while (> pos 0)
    ;; Variables are changed simultaneously by 'psetf', so there's no need for a temporary variable.
    do (psetf
         pos (+ (* pos 2) (- prev-pos) (* acc dt dt))
         prev-pos pos)
    finally (return (list time vel)))) 
```

现在，假设我们实际上需要速度来计算下一个时间步。在这种情况下，我们简单地不能使用上面的近似，而需要使用*速度 Verlet*算法。

# 速度 Verlet

在某些方面，这个算法甚至比上面的更简单。我们可以这样计算一切

这实际上是上面的运动学方程，每个时间步都在求解，，和。你也可以这样拆分方程

这里是速度 Verlet 方法的代码：

```
function velocity_verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0
    vel = 0.0

    while (pos > 0.0)
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
    end

    return time, vel
end 
```

```
timestep velocity_verlet(double pos, double acc, double dt) {

  double time = 0;
  double vel = 0;
  while (pos > 0) {
    time += dt;
    pos += vel * dt + 0.5 * acc * dt * dt;
    vel += acc * dt;
  }

  return timestep { time, vel };
} 
```

```
void velocity_verlet(double *time, double *vel,
                     double pos, double acc, double dt) {
    *vel = 0.0;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        pos += (*vel) * dt + 0.5 * acc * dt * dt;
        *vel += acc * dt;
    }
} 
```

```
static VerletValues velocity_verlet(double pos, double acc, double dt) {

    // Note that we are using a temp variable for the previous position
    double time, vel;
    vel = 0;
    time = 0;
    while (pos > 0) {
        time += dt;
        pos += vel*dt + 0.5*acc * dt * dt;
        vel += acc*dt;
    }
    return new VerletValues(time, vel);
} 
```

```
def velocity_verlet(pos, acc, dt):
    time = 0
    vel = 0

    while pos > 0:
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt
        vel += acc * dt

    return time, vel 
```

```
velocityVerlet :: Method
velocityVerlet acc dt (xOld, _, aOld, _) (x, v, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = v + 0.5 * (aOld + a) * dt
    a' = acc (x', v', a, t + dt) 
```

```
function velocityVerlet(pos, acc, dt) {
  let time = 0;
  let vel = 0;

  while (pos > 0) {
    time += dt;
    pos += vel * dt + 0.5 * acc * dt * dt;
    vel += acc * dt;
  }

  return { time, vel };
} 
```

```
fn velocity_verlet(mut pos: f64, acc: f64, dt: f64) -> (f64, f64) {
    let mut time = 0.0;
    let mut vel = 0.0;

    while pos > 0.0 {
        time += dt;
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
    }

    (time, vel)
} 
```

```
func velocityVerlet(pos: Double, acc: Double, dt: Double) -> (time: Double, vel: Double) {
    var pos = pos
    var time, vel : Double
    vel = 0
    time = 0

    while (pos > 0) {
        time += dt
        pos += vel*dt + 0.5*acc * dt * dt
        vel += acc*dt
    }

    return (time:time, vel:vel)
} 
```

```
SUBROUTINE velocity_verlet(pos, acc, dt, time, vel) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel

    time     = 0d0
    vel      = 0d0

    DO
        IF (pos > 0d0) THEN
            time = time + dt
            pos  = pos + vel * dt + 0.5d0 * acc * dt ** 2 
            vel  = vel + acc * dt
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE velocity_verlet 
```

```
def velocity_verlet(pos, acc, dt)

    vel = 0
    time = 0
    while pos > 0 do
        time += dt
        pos += vel*dt + 0.5*acc * dt * dt
        vel += acc*dt
    end

   return time, vel

end 
```

```
func velocityVerlet(pos, acc, dt float64) (time, vel float64) {
    time, vel = 0, 0

    for pos > 0 {
        time += dt
        pos += vel*dt + .5*acc*dt*dt
        vel += acc * dt
    }

    return
} 
```

```
# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
# RET xmm1 - velocity
velocity_verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds the velocity value
  pxor   xmm4, xmm4                  # Holds the time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    velocity_verlet_return
  movsd  xmm5, half                  # xmm5 = 0.5 * dt * dt * acc
  mulsd  xmm5, xmm2
  mulsd  xmm5, xmm2
  mulsd  xmm5, xmm1
velocity_verlet_loop:
  movsd  xmm6, xmm3                  # Move velocity into register
  mulsd  xmm6, xmm2                  # Calculate new position
  addsd  xmm6, xmm5
  addsd  xmm0, xmm6
  addsd  xmm4, xmm2                  # Incrementing time
  movsd  xmm3, xmm4                  # Updating velocity
  mulsd  xmm3, xmm1
  comisd xmm0, xmm7
  ja     velocity_verlet_loop
velocity_verlet_return:
  movsd  xmm0, xmm4                  # Saving time and velocity
  movsd  xmm1, xmm3
  ret 
```

```
fun velocityVerlet(_pos: Double, acc: Double, dt: Double): VerletValues {
    var pos = _pos
    var time = 0.0
    var vel = 0.0
    while (pos > 0) {
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt
        vel += acc * dt
    }
    return VerletValues(time, vel)
} 
```

```
func velocityVerlet(pos_in, acc, dt: float): (float, float) =
  var
    pos: float = pos_in
    time: float = 0.0
    vel: float = 0.0

  while pos > 0.0:
    time += dt
    pos += vel * dt + 0.5 * acc * dt * dt
    vel += acc * dt

  (time, vel) 
```

```
(defun velocity-verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using the velocity in calculations."
  (loop
    for time = 0 then (incf time dt)
    for vel = 0 then (incf vel (* acc dt))
    for p = pos then (incf p (+ (* vel dt) (* 0.5 acc dt dt)))
    while (> p 0)
    finally (return (list time vel)))) 
```

尽管这种方法比上面提到的简单 Verlet 方法更广泛使用，但它不幸地有一个误差项，其大小是前者的两个数量级。话虽如此，如果你想要一个有许多相互依赖对象的模拟——比如重力模拟——速度 Verlet 算法是一个方便的选择；然而，你可能需要进一步玩弄技巧以使所有内容适当地缩放。这类模拟有时被称为*n-body*模拟，其中一种技巧是 Barnes-Hut 算法，它将 n-body 模拟的复杂度从降低到。

## 视频解释

这里有一个描述 Verlet 积分的视频：

[`www.youtube-nocookie.com/embed/g55QvpAev0I`](https://www.youtube-nocookie.com/embed/g55QvpAev0I)

## 示例代码

这两种方法都是通过逐时间步迭代来工作的，并且可以用任何语言直接编写。为了参考，这里有一些代码片段，它们使用了经典和速度 Verlet 方法来找到从给定高度掉落的小球击中地面所需的时间。

```
function verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0

    while (pos > 0)
        time += dt
        temp_pos = pos
        pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    end

    return time
end

function stormer_verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0
    vel = 0.0

    while (pos > 0.0)
        time += dt
        temp_pos = pos
        pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        # Because acceleration is constant, velocity is straightforward
        vel += acc * dt
    end

    return time, vel
end

function velocity_verlet(pos::Float64, acc::Float64, dt::Float64)
    prev_pos = pos
    time = 0.0
    vel = 0.0

    while (pos > 0.0)
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
    end

    return time, vel
end

function main()
    time = verlet(5.0, -10.0, 0.01);
    println("[#]\nTime for Verlet integration is:")
    println("$(time)")

    time, vel = stormer_verlet(5.0, -10.0, 0.01);
    println("[#]\nTime for Stormer Verlet integration is:")
    println("$(time)")
    println("[#]\nVelocity for Stormer Verlet integration is:")
    println("$(vel)")

    time, vel = velocity_verlet(5.0, -10.0, 0.01);
    println("[#]\nTime for velocity Verlet integration is:")
    println("$(time)")
    println("[#]\nVelocity for velocity Verlet integration is:")
    println("$(vel)")

end

main() 
```

```
#include <iomanip>
#include <iostream>

struct timestep {
  double time;
  double vel;
};

double verlet(double pos, double acc, double dt) {

  double prev_pos = pos;
  double time = 0;

  while (pos > 0) {
    time += dt;
    double next_pos = pos * 2 - prev_pos + acc * dt * dt;
    prev_pos = pos;
    pos = next_pos;
  }

  return time;
}

timestep stormer_verlet(double pos, double acc, double dt) {

  double prev_pos = pos;
  double time = 0;
  double vel = 0;
  while (pos > 0) {
    time += dt;
    double next_pos = pos * 2 - prev_pos + acc * dt * dt;
    prev_pos = pos;
    pos = next_pos;

    // The acceleration is constant, so the velocity is
    // straightforward
    vel += acc * dt;
  }

  return timestep { time, vel };
}

timestep velocity_verlet(double pos, double acc, double dt) {

  double time = 0;
  double vel = 0;
  while (pos > 0) {
    time += dt;
    pos += vel * dt + 0.5 * acc * dt * dt;
    vel += acc * dt;
  }

  return timestep { time, vel };
}

int main() {
  std::cout << std::fixed << std::setprecision(8);

  // Note that depending on the simulation, you might want to have the
  // Verlet loop outside.

  // For example, if your acceleration chages as a function of time,
  // you might need to also change the acceleration to be read into
  // each of these functions.

  double time = verlet(5.0, -10, 0.01);
  std::cout << "[#]\nTime for Verlet integration is:\n" \
            << time << std::endl;

  timestep timestep_sv = stormer_verlet(5.0, -10, 0.01);
  std::cout << "[#]\nTime for Stormer Verlet integration is:\n" \
            << timestep_sv.time << std::endl;
  std::cout << "[#]\nVelocity for Stormer Verlet integration is:\n" \
            << timestep_sv.vel << std::endl;

  timestep timestep_vv = velocity_verlet(5.0, -10, 0.01);
  std::cout << "[#]\nTime for velocity Verlet integration is:\n" \
            << timestep_vv.time << std::endl;
  std::cout << "[#]\nVelocity for velocity Verlet integration is:\n" \
            << timestep_vv.vel << std::endl;

  return 0;

} 
```

```
#include <stdio.h>

void verlet(double *time, double pos, double acc, double dt) {
    double prev_pos, temp_pos;
    prev_pos = pos;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        temp_pos = pos;
        pos = pos * 2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;
    }
}

void stormer_verlet(double *time, double *vel,
                    double pos, double acc, double dt) {
    double prev_pos, temp_pos;
    prev_pos = pos;
    *vel = 0.0;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        temp_pos = pos;
        pos = pos * 2 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;

        *vel += acc * dt;
    }
}

void velocity_verlet(double *time, double *vel,
                     double pos, double acc, double dt) {
    *vel = 0.0;
    *time = 0.0;

    while (pos > 0) {
        *time += dt;
        pos += (*vel) * dt + 0.5 * acc * dt * dt;
        *vel += acc * dt;
    }
}

int main() {
    double time, vel;

    verlet(&time, 5.0, -10, 0.01);
    printf("[#]\nTime for Verlet integration is:\n");
    printf("%lf\n", time);

    stormer_verlet(&time, &vel, 5.0, -10, 0.01);
    printf("[#]\nTime for Stormer Verlet integration is:\n");
    printf("%lf\n", time);
    printf("[#]\nVelocity for Stormer Verlet integration is:\n");
    printf("%lf\n", vel);

    velocity_verlet(&time, &vel, 5.0, -10, 0.01);
    printf("[#]\nTime for velocity Verlet integration is:\n");
    printf("%lf\n", time);
    printf("[#]\nVelocity for Stormer Verlet integration is:\n");
    printf("%lf\n", vel);

    return 0;
} 
```

```
public class Verlet {

    private static class VerletValues {
        public double time;
        public double vel;

        public VerletValues(double time, double vel) {
            this.time = time;
            this.vel = vel;
        }
    }

    static double verlet(double pos, double acc, double dt) {

      // Note that we are using a temp variable for the previous position
      double prev_pos, temp_pos, time;
      prev_pos = pos;
      time = 0;

      while (pos > 0) {
            time += dt;
            temp_pos = pos;
            pos = pos*2 - prev_pos + acc * dt * dt;
            prev_pos = temp_pos;
        }

        return time;
    }

    static VerletValues stormer_verlet(double pos, double acc, double dt) {

        // Note that we are using a temp variable for the previous position
        double prev_pos, temp_pos, time, vel;
        prev_pos = pos;
        vel = 0;
        time = 0;
        while (pos > 0) {
            time += dt;
            temp_pos = pos;
            pos = pos*2 - prev_pos + acc * dt * dt;
            prev_pos = temp_pos;

            // The acceleration is constant, so the velocity is straightforward
             vel += acc*dt;
        }

       return new VerletValues(time, vel);
    }

    static VerletValues velocity_verlet(double pos, double acc, double dt) {

        // Note that we are using a temp variable for the previous position
        double time, vel;
        vel = 0;
        time = 0;
        while (pos > 0) {
            time += dt;
            pos += vel*dt + 0.5*acc * dt * dt;
            vel += acc*dt;
        }
        return new VerletValues(time, vel);
    }

    public static void main(String[] args) {

        double verletTime = verlet(5.0, -10, 0.01);
        System.out.println("[#]\nTime for Verlet integration is:");
        System.out.println(verletTime);

        VerletValues stormerVerlet = stormer_verlet(5.0, -10, 0.01);
        System.out.println("[#]\nTime for Stormer Verlet integration is:");
        System.out.println(stormerVerlet.time);
        System.out.println("[#]\nVelocity for Stormer Verlet integration is:");
        System.out.println(stormerVerlet.vel);

        VerletValues velocityVerlet = velocity_verlet(5.0, -10, 0.01);
        System.out.println("[#]\nTime for velocity Verlet integration is:");
        System.out.println(velocityVerlet.time);
        System.out.println("[#]\nVelocity for velocity Verlet integration is:");
        System.out.println(velocityVerlet.vel);

    }
} 
```

```
def verlet(pos, acc, dt):
    prev_pos = pos
    time = 0

    while pos > 0:
        time += dt
        next_pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos, pos = pos, next_pos

    return time

def stormer_verlet(pos, acc, dt):
    prev_pos = pos
    time = 0
    vel = 0

    while pos > 0:
        time += dt
        next_pos = pos * 2 - prev_pos + acc * dt * dt
        prev_pos, pos = pos, next_pos
        vel += acc * dt

    return time, vel

def velocity_verlet(pos, acc, dt):
    time = 0
    vel = 0

    while pos > 0:
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt
        vel += acc * dt

    return time, vel

def main():
    time = verlet(5, -10, 0.01)
    print("[#]\nTime for Verlet integration is:")
    print("{:.10f}".format(time))

    time, vel = stormer_verlet(5, -10, 0.01)
    print("[#]\nTime for Stormer Verlet integration is:")
    print("{:.10f}".format(time))
    print("[#]\nVelocity for Stormer Verlet integration is:")
    print("{:.10f}".format(vel))

    time, vel = velocity_verlet(5, -10, 0.01)
    print("[#]\nTime for velocity Verlet integration is:")
    print("{:.10f}".format(time))
    print("[#]\nVelocity for velocity Verlet integration is:")
    print("{:.10f}".format(vel))

if __name__ == '__main__':
    main() 
```

```
-- submitted by Jie
type Time = Double

type Position = Double

type Speed = Double

type Acceleration = Double

type Particle = (Position, Speed, Acceleration, Time)

type Model = Particle -> Acceleration

type Method = Model -> Time -> Particle -> Particle -> Particle

verlet :: Method
verlet acc dt (xOld, _, _, _) (x, _, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = 0
    a' = acc (x', v', a, t + dt)

stormerVerlet :: Method
stormerVerlet acc dt (xOld, _, _, _) (x, _, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = (x' - x) / dt
    a' = acc (x', v', a, t + dt)

velocityVerlet :: Method
velocityVerlet acc dt (xOld, _, aOld, _) (x, v, a, t) = (x', v', a', t + dt)
  where
    x' = 2 * x - xOld + a * dt ^ 2
    v' = v + 0.5 * (aOld + a) * dt
    a' = acc (x', v', a, t + dt)

trajectory :: Method -> Model -> Time -> Particle -> [Particle]
trajectory method acc dt p0@(x, v, a, t0) = traj
  where
    traj = p0 : p1 : zipWith (method acc dt) traj (tail traj)
    p1 = (x', v', acc (x', v', a, t0 + dt), t0 + dt)
    x' = x + v * dt + 0.5 * a * dt ^ 2
    v' = v + a * dt

main :: IO ()
main = do
  let p0 = (5, 0, -10, 0)
      dt = 0.001
      freefall _ = -10
      aboveGround (x, _, _, _) = x > 0
      timeVelocity m =
        let (_, v, _, t) = last $ takeWhile aboveGround $ trajectory m freefall dt p0
         in (show t, show v)

  putStrLn "[#]\nTime for Verlet integration is:"
  putStrLn $ fst $ timeVelocity verlet
  putStrLn "[#]\nTime for Stormer Verlet integration is:"
  putStrLn $ fst $ timeVelocity stormerVerlet
  putStrLn "[#]\nVelocity for Stormer Verlet integration is:"
  putStrLn $ snd $ timeVelocity stormerVerlet
  putStrLn "[#]\nTime for velocity Verlet integration is:"
  putStrLn $ fst $ timeVelocity velocityVerlet
  putStrLn "[#]\nVelocity for velocity Verlet integration is:"
  putStrLn $ snd $ timeVelocity velocityVerlet 
```

```
function verlet(pos, acc, dt) {
  let prevPos = pos;
  let time = 0;
  let tempPos;

  while (pos > 0) {
    time += dt;
    tempPos = pos;
    pos = pos * 2 - prevPos + acc * dt * dt;
    prevPos = tempPos;
  }

  return time;
}

function stormerVerlet(pos, acc, dt) {
  let prevPos = pos;
  let time = 0;
  let vel = 0;
  let tempPos;

  while (pos > 0) {
    time += dt;
    tempPos = pos;
    pos = pos * 2 - prevPos + acc * dt * dt;
    prevPos = tempPos;

    vel += acc * dt;
  }

  return { time, vel };
}

function velocityVerlet(pos, acc, dt) {
  let time = 0;
  let vel = 0;

  while (pos > 0) {
    time += dt;
    pos += vel * dt + 0.5 * acc * dt * dt;
    vel += acc * dt;
  }

  return { time, vel };
}

const time = verlet(5, -10, 0.01);
console.log(`[#]\nTime for Verlet integration is:`);
console.log(`${time}`);

const stormer = stormerVerlet(5, -10, 0.01);
console.log(`[#]\nTime for Stormer Verlet integration is:`);
console.log(`${stormer.time}`);
console.log(`[#]\nVelocity for Stormer Verlet integration is:`);
console.log(`${stormer.vel}`);

const velocity = velocityVerlet(5, -10, 0.01);
console.log(`[#]\nTime for velocity Verlet integration is:`);
console.log(`${velocity.time}`);
console.log(`[#]\nVelocity for velocity Verlet integration is:`);
console.log(`${velocity.vel}`); 
```

```
fn verlet(mut pos: f64, acc: f64, dt: f64) -> f64 {
    let mut prev_pos = pos;
    let mut time = 0.0;

    while pos > 0.0 {
        time += dt;
        let temp_pos = pos;
        pos = pos * 2.0 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;
    }

    time
}

fn stormer_verlet(mut pos: f64, acc: f64, dt: f64) -> (f64, f64) {
    let mut prev_pos = pos;
    let mut time = 0.0;
    let mut vel = 0.0;

    while pos > 0.0 {
        time += dt;
        let temp_pos = pos;
        pos = pos * 2.0 - prev_pos + acc * dt * dt;
        prev_pos = temp_pos;

        // Because acceleration is constant, velocity is
        // straightforward
        vel += acc * dt;
    }

    (time, vel)
}

fn velocity_verlet(mut pos: f64, acc: f64, dt: f64) -> (f64, f64) {
    let mut time = 0.0;
    let mut vel = 0.0;

    while pos > 0.0 {
        time += dt;
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
    }

    (time, vel)
}

fn main() {
    let time_v = verlet(5.0, -10.0, 0.01);
    let (time_sv, vel_sv) = stormer_verlet(5.0, -10.0, 0.01);
    let (time_vv, vel_vv) = velocity_verlet(5.0, -10.0, 0.01);

    println!("[#]\nTime for Verlet integration is:");
    println!("{}", time_v);

    println!("[#]\nTime for Stormer Verlet integration is:");
    println!("{}", time_sv);
    println!("[#]\nVelocity for Stormer Verlet integration is:");
    println!("{}", vel_sv);

    println!("[#]\nTime for velocity Verlet integration is:");
    println!("{}", time_vv);
    println!("[#]\nVelocity for velocity Verlet integration is:");
    println!("{}", vel_vv);
} 
```

```
func verlet(pos: Double, acc: Double, dt: Double) -> Double {
    var pos = pos
    var temp_pos, time: Double
    var prev_pos = pos
    time = 0.0

    while (pos > 0) {
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    }

    return time
}

func stormerVerlet(pos: Double, acc: Double, dt: Double) -> (time: Double, vel: Double) {
    var pos = pos
    var temp_pos, time, vel: Double
    var prev_pos = pos
    vel = 0
    time = 0

    while (pos > 0) {
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        vel += acc*dt
    }

    return (time:time, vel:vel)
}

func velocityVerlet(pos: Double, acc: Double, dt: Double) -> (time: Double, vel: Double) {
    var pos = pos
    var time, vel : Double
    vel = 0
    time = 0

    while (pos > 0) {
        time += dt
        pos += vel*dt + 0.5*acc * dt * dt
        vel += acc*dt
    }

    return (time:time, vel:vel)
}

func main() {
    let verletTime = verlet(pos: 5.0, acc: -10.0, dt: 0.01)
    print("[#]\nTime for Verlet integration is:")
    print("\(verletTime)")

    let stormer = stormerVerlet(pos: 5.0, acc: -10.0, dt: 0.01);
    print("[#]\nTime for Stormer Verlet integration is:")
    print("\(stormer.time)")
    print("[#]\nVelocity for Stormer Verlet integration is:")
    print("\(stormer.vel)")

    let velVerlet = velocityVerlet(pos: 5.0, acc: -10, dt: 0.01)
    print("[#]\nTime for velocity Verlet integration is:")
    print("\(velVerlet.time)")
    print("[#]\nVelocity for velocity Verlet integration is:")
    print("\(velVerlet.vel)")
}

main() 
```

```
SUBROUTINE verlet(pos, acc, dt, time) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time
    REAL(8)                :: prev_pos, next_pos

    prev_pos = pos
    time     = 0d0

    DO
        IF (pos > 0d0) THEN
            time     = time + dt
            next_pos = pos * 2d0 - prev_pos + acc * dt ** 2
            prev_pos = pos
            pos      = next_pos
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE verlet

SUBROUTINE stormer_verlet(pos, acc, dt, time, vel) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel
    REAL(8)                :: prev_pos, next_pos

    prev_pos = pos 
    time     = 0d0
    vel      = 0d0

    DO
        IF (pos > 0d0) THEN
            time     = time + dt
            next_pos = pos * 2 - prev_pos + acc * dt ** 2
            prev_pos = pos
            pos      = next_pos
            vel      = vel + acc * dt
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE stormer_verlet 

SUBROUTINE velocity_verlet(pos, acc, dt, time, vel) 
    IMPLICIT NONE
    REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel

    time     = 0d0
    vel      = 0d0

    DO
        IF (pos > 0d0) THEN
            time = time + dt
            pos  = pos + vel * dt + 0.5d0 * acc * dt ** 2 
            vel  = vel + acc * dt
        ELSE
            EXIT
        END IF
    END DO
END SUBROUTINE velocity_verlet 

PROGRAM verlet_integration

    IMPLICIT NONE 
    REAL(8) :: pos,acc, dt, time, vel

    INTERFACE
        SUBROUTINE verlet(pos, acc, dt, time)
        REAL(8), INTENT(INOUT) :: pos, acc, dt, time
        REAL(8)                :: prev_pos, next_pos
        END SUBROUTINE
    END INTERFACE 

    INTERFACE 
        SUBROUTINE stormer_verlet(pos, acc, dt, time, vel) 
            REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel
            REAL(8)                :: prev_pos, next_pos
        END SUBROUTINE 
    END INTERFACE 

    INTERFACE 
        SUBROUTINE velocity_verlet(pos, acc, dt, time, vel) 
            REAL(8), INTENT(INOUT) :: pos, acc, dt, time, vel
            REAL(8)                :: prev_pos, next_pos 
        END SUBROUTINE 
    END INTERFACE 

    pos = 5d0
    acc = -10d0
    dt  = 0.01d0
    ! Verlet 
    CALL verlet(pos, acc, dt, time)

    WRITE(*,*) '[#]'
    WRITE(*,*) 'Time for Verlet integration:'
    WRITE(*,*) time 

    ! stormer Verlet 
    pos = 5d0
    CALL stormer_verlet(pos, acc, dt, time, vel)

    WRITE(*,*) '[#]'
    WRITE(*,*) 'Time for Stormer Verlet integration:'
    WRITE(*,*) time
    WRITE(*,*) '[#]'
    WRITE(*,*) 'Velocity for Stormer Verlet integration:'
    WRITE(*,*) vel

    ! Velocity Verlet
    pos = 5d0
    CALL velocity_verlet(pos, acc, dt, time, vel)

    WRITE(*,*) '[#]'
    WRITE(*,*) 'Time for velocity Verlet integration:'
    WRITE(*,*) time
    WRITE(*,*) '[#]'
    WRITE(*,*) 'Velocity for velocity Verlet integration:'
    WRITE(*,*) vel

END PROGRAM verlet_integration 
```

```
def verlet(pos, acc, dt)

    prev_pos = pos
    time = 0
    while pos > 0 do
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos
    end

   return time

end

def stormer_verlet(pos, acc, dt)

    prev_pos = pos
    vel = 0
    time = 0
    while pos > 0 do
        time += dt
        temp_pos = pos
        pos = pos*2 - prev_pos + acc * dt * dt
        prev_pos = temp_pos

        vel += acc*dt
    end

   return time, vel

end

def velocity_verlet(pos, acc, dt)

    vel = 0
    time = 0
    while pos > 0 do
        time += dt
        pos += vel*dt + 0.5*acc * dt * dt
        vel += acc*dt
    end

   return time, vel

end

puts "[#]\nTime for Verlet integration is:"
p verlet(5.0, -10, 0.01)

time, vel = stormer_verlet(5.0, -10, 0.01)
puts "[#]\nTime for Stormer Verlet integration is:"
p time
puts "[#]\nVelocity for Stormer Verlet integration is:"
p vel

time, vel = velocity_verlet(5.0, -10, 0.01)
puts "[#]\nTime for velocity Verlet integration is:"
p time
puts "[#]\nVelocity for velocity Verlet integration is:"
p vel 
```

```
package main

import "fmt"

func verlet(pos, acc, dt float64) (time float64) {
    prevPos := pos
    time = 0

    for pos > 0 {
        time += dt
        nextPos := pos*2 - prevPos + acc*dt*dt
        prevPos, pos = pos, nextPos
    }

    return
}

func stormerVerlet(pos, acc, dt float64) (time, vel float64) {
    prevPos := pos
    time, vel = 0, 0

    for pos > 0 {
        time += dt
        vel += acc * dt
        nextPos := pos*2 - prevPos + acc*dt*dt
        prevPos, pos = pos, nextPos
    }

    return
}

func velocityVerlet(pos, acc, dt float64) (time, vel float64) {
    time, vel = 0, 0

    for pos > 0 {
        time += dt
        pos += vel*dt + .5*acc*dt*dt
        vel += acc * dt
    }

    return
}

func main() {
    time := verlet(5., -10., .01)
    fmt.Println("[#]\nTime for Verlet integration is:")
    fmt.Println(time)

    time, vel := stormerVerlet(5., -10., .01)
    fmt.Println("[#]\nTime for Stormer Verlet integration is:")
    fmt.Println(time)
    fmt.Println("[#]\nVelocity for Stormer Verlet integration is:")
    fmt.Println(vel)

    time, vel = velocityVerlet(5., -10., .01)
    fmt.Println("[#]\nTime for velocity Verlet integration is:")
    fmt.Println(time)
    fmt.Println("[#]\nVelocity for velocity Verlet integration is:")
    fmt.Println(vel)
} 
```

```
.intel_syntax noprefix

.section .rodata
  zero:          .double 0.0
  two:           .double 2.0
  half:          .double 0.5
  verlet_fmt:    .string "[#]\nTime for Verlet integration is:\n%lf\n"
  stormer_fmt:   .string "[#]\nTime for Stormer Verlet Integration is:\n%lf\n[#]\nVelocity for Stormer Verlet Integration is:\n%lf\n"
  velocity_fmt:  .string "[#]\nTime for Velocity Verlet Integration is:\n%lf\n[#]\nVelocity for Velocity Verlet Integration is:\n%lf\n"
  pos:           .double 5.0
  acc:           .double -10.0
  dt:            .double 0.01

.section .text
  .global main
  .extern printf

# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    verlet_return
  movsd  xmm6, xmm1                  # xmm6 = acc * dt * dt
  mulsd  xmm6, xmm2
  mulsd  xmm6, xmm2
  movsd  xmm5, xmm0                  # Holds previous position
verlet_loop:
  addsd  xmm3, xmm2                  # Adding dt to time
  movsd  xmm4, xmm0                  # Hold old value of posistion
  addsd  xmm0, xmm0                  # Calculating new position
  subsd  xmm0, xmm5
  addsd  xmm0, xmm6
  movsd  xmm5, xmm4
  comisd xmm0, xmm7                  # Check if position is greater then 0.0
  ja     verlet_loop
verlet_return:
  movsd  xmm0, xmm3                  # Saving time value
  ret

# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
# RET xmm1 - velocity
stormer_verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    stormer_verlet_return
  movsd  xmm6, xmm1                  # xmm6 = acc * dt * dt
  mulsd  xmm6, xmm2
  mulsd  xmm6, xmm2
  movsd  xmm5, xmm0                  # Holds previous position
stormer_verlet_loop:
  addsd  xmm3, xmm2                  # Adding dt to time
  movsd  xmm4, xmm0                  # Hold old value of posistion
  addsd  xmm0, xmm0                  # Calculating new position
  subsd  xmm0, xmm5
  addsd  xmm0, xmm6
  movsd  xmm5, xmm4
  comisd xmm0, xmm7                  # Check if position is greater then 0.0
  ja     stormer_verlet_loop
stormer_verlet_return:
  movsd  xmm0, xmm3                  # Saving time and velocity
  mulsd  xmm3, xmm1
  movsd  xmm1, xmm3
  ret

# xmm0 - pos
# xmm1 - acc
# xmm2 - dt
# RET xmm0 - time
# RET xmm1 - velocity
velocity_verlet:
  pxor   xmm7, xmm7                  # Holds 0 for comparisons
  pxor   xmm3, xmm3                  # Holds the velocity value
  pxor   xmm4, xmm4                  # Holds the time value
  comisd xmm0, xmm7                  # Check if pos is greater then 0.0
  jbe    velocity_verlet_return
  movsd  xmm5, half                  # xmm5 = 0.5 * dt * dt * acc
  mulsd  xmm5, xmm2
  mulsd  xmm5, xmm2
  mulsd  xmm5, xmm1
velocity_verlet_loop:
  movsd  xmm6, xmm3                  # Move velocity into register
  mulsd  xmm6, xmm2                  # Calculate new position
  addsd  xmm6, xmm5
  addsd  xmm0, xmm6
  addsd  xmm4, xmm2                  # Incrementing time
  movsd  xmm3, xmm4                  # Updating velocity
  mulsd  xmm3, xmm1
  comisd xmm0, xmm7
  ja     velocity_verlet_loop
velocity_verlet_return:
  movsd  xmm0, xmm4                  # Saving time and velocity
  movsd  xmm1, xmm3
  ret

main:
  push   rbp
  movsd  xmm0, pos                   # Calling verlet
  movsd  xmm1, acc
  movsd  xmm2, dt
  call   verlet
  mov    rdi, OFFSET verlet_fmt      # Print output
  mov    rax, 1
  call   printf
  movsd  xmm0, pos                   # Calling stormer_verlet
  movsd  xmm1, acc
  movsd  xmm2, dt
  call   stormer_verlet
  mov    rdi, OFFSET stormer_fmt     # Print output
  mov    rax, 1
  call   printf
  movsd  xmm0, pos                   # Calling velocity_verlet
  movsd  xmm1, acc
  movsd  xmm2, dt
  call   velocity_verlet
  mov    rdi, OFFSET velocity_fmt    # Print output
  mov    rax, 1
  call   printf
  pop    rbp
  xor    rax, rax                      # Set exit code to 0
  ret 
```

```
data class VerletValues(val time: Double, val vel: Double)

fun verlet(_pos: Double, acc: Double, dt: Double): Double {
    var pos = _pos  // Since function parameter are val and can't be modified
    var prevPos = pos
    var time = 0.0

    while (pos > 0) {
        time += dt
        val nextPos = pos * 2 - prevPos + acc * dt * dt
        prevPos = pos
        pos = nextPos
    }
    return time
}

fun stormerVerlet(_pos: Double, acc: Double, dt: Double): VerletValues {
    var pos = _pos
    var prevPos = pos
    var time = 0.0
    var vel = 0.0
    while (pos > 0) {
        time += dt
        val nextPos = pos * 2 - prevPos + acc * dt * dt
        prevPos = pos
        pos = nextPos
        vel += acc * dt
    }
    return VerletValues(time, vel)
}

fun velocityVerlet(_pos: Double, acc: Double, dt: Double): VerletValues {
    var pos = _pos
    var time = 0.0
    var vel = 0.0
    while (pos > 0) {
        time += dt
        pos += vel * dt + 0.5 * acc * dt * dt
        vel += acc * dt
    }
    return VerletValues(time, vel)
}

fun main(args: Array<String>) {
    val verletTime = verlet(5.0, -10.0, 0.01)
    println("[#]\nTime for Verlet integration is:")
    println("$verletTime")

    val stormerVerlet = stormerVerlet(5.0, -10.0, 0.01)
    println("[#]\nTime for Stormer Verlet integration is:")
    println("${stormerVerlet.time}")
    println("[#]\nVelocity for Stormer Verlet integration is:")
    println("${stormerVerlet.vel}")

    val velocityVerlet = velocityVerlet(5.0, -10.0, 0.01)
    println("[#]\nTime for Velocity Verlet integration is:")
    println("${velocityVerlet.time}")
    println("[#]\nVelocity for Velocity Verlet integration is:")
    println("${velocityVerlet.vel}")
} 
```

```
func verlet(pos_in, acc, dt: float): float =
  var
    pos: float = pos_in
    prevPos: float = pos
    time: float = 0.0
    tempPos: float

  while pos > 0.0:
    time += dt
    tempPos = pos
    pos = pos * 2 - prevPos + acc * dt * dt
    prevPos = tempPos

  time

func stormerVerlet(pos_in, acc, dt: float): (float, float) =
  var
    pos: float = pos_in
    prevPos: float = pos
    time: float = 0.0
    vel: float = 0.0
    tempPos: float

  while pos > 0.0:
    time += dt
    tempPos = pos
    pos = pos * 2 - prevPos + acc * dt * dt
    prevPos = tempPos

    vel += acc * dt

  (time, vel)

func velocityVerlet(pos_in, acc, dt: float): (float, float) =
  var
    pos: float = pos_in
    time: float = 0.0
    vel: float = 0.0

  while pos > 0.0:
    time += dt
    pos += vel * dt + 0.5 * acc * dt * dt
    vel += acc * dt

  (time, vel)

when isMainModule:
  let timeV = verlet(5.0, -10.0, 0.01)
  echo "[#]\nTime for Verlet integration is:"
  echo timeV

  let (timeSV, velSV) = stormerVerlet(5.0, -10.0, 0.01)
  echo "[#]\nTime for Stormer Verlet integration is:"
  echo timeSV
  echo "[#]\nVelocity for Stormer Verlet integration is:"
  echo velSV

  let (timeVV, velVV) = velocityVerlet(5.0, -10.0, 0.01)
  echo "[#]\nTime for velocity Verlet integration is:"
  echo timeVV
  echo "[#]\nVelocity for velocity Verlet integration is:"
  echo velVV 
```

```
;;;; Verlet integration implementation in Common Lisp

(defun verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using Verlet integration."
  (loop
    with prev-pos = pos
    for time = 0 then (incf time dt)
    while (> pos 0)
    ;; The starting speed is assumed to be zero.
    do (psetf
         pos (+ (* pos 2) (- prev-pos) (* acc dt dt))
         prev-pos pos)
    finally (return time)))

(defun stormer-verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using the Stormer-Verlet method."
  (loop
    with prev-pos = pos
    for time = 0 then (incf time dt)
    for vel = 0 then (incf vel (* acc dt))
    while (> pos 0)
    ;; Variables are changed simultaneously by 'psetf', so there's no need for a temporary variable.
    do (psetf
         pos (+ (* pos 2) (- prev-pos) (* acc dt dt))
         prev-pos pos)
    finally (return (list time vel))))

(defun velocity-verlet (pos acc dt)
  "Integrates Newton's equation for motion while pos > 0 using the velocity in calculations."
  (loop
    for time = 0 then (incf time dt)
    for vel = 0 then (incf vel (* acc dt))
    for p = pos then (incf p (+ (* vel dt) (* 0.5 acc dt dt)))
    while (> p 0)
    finally (return (list time vel))))

(format T "[#]~%Time for Verlet integration:~%")
(format T "~d~%" (verlet 5 -10 0.01))

(defvar stormer-verlet-result (stormer-verlet 5 -10 0.01))
(format T "[#]~%Time for Stormer Verlet integration is:~%")
(format T "~d~%" (first stormer-verlet-result))
(format T "[#]~%Velocity for Stormer Verlet integration is:~%")
(format T "~d~%" (second stormer-verlet-result))

(defvar velocity-verlet-result (velocity-verlet 5 -10 0.01))
(format T "[#]~%Time for velocity Verlet integration is:~%")
(format T "~d~%" (first velocity-verlet-result))
(format T "[#]~%Velocity for velocity Verlet integration is:~%")
(format T "~d~%" (second velocity-verlet-result)) 
```

## 许可证

##### 代码示例

代码示例受 MIT 许可协议保护（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并受[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)许可。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 提交请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下提交请求已修改本章的文本或图形：

+   无
