# Per-environment settings shared by the queue_*.sh scripts. Expects $ENV to be set.
#   ENV_ID     gym id
#   EQ_ARGS    equilibrium observation (Lyapunov methods only)
#   OFF_STEPS  total timesteps for SAC / LSAC (empty = not run on this env)
#   ON_STEPS   total timesteps for PPO / LPPO / LY
#   ON_ARGS    extra PPO-family arguments
case "$ENV" in
    pendulum)
        ENV_ID=Pendulum-v1
        EQ_ARGS="--eq-obs 1 0 0"  # obs = [cos theta, sin theta, theta_dot]; upright is (1, 0, 0)
        OFF_STEPS=100000
        ON_STEPS=100000
        ON_ARGS=""
        ;;
    double)
        ENV_ID=InvertedDoublePendulum-v5
        EQ_ARGS="--eq-obs 0 0 0 1 1 0 0 0 0"  # obs = [x, sin t1, sin t2, cos t1, cos t2, v, w1, w2, F_c]
        OFF_STEPS=500000
        ON_STEPS=500000
        ON_ARGS=""
        ;;
    quad)
        ENV_ID=Quadrotor-Still-v1
        EQ_ARGS=""  # obs is the error from the reference trajectory; equilibrium is all zeros
        OFF_STEPS=""
        ON_STEPS=15000000
        ON_ARGS="--num-envs 8 --ent-coef 0.001"
        ;;
    hopper)
        ENV_ID=Hopper-v5
        EQ_ARGS=""  # full-state equilibrium is not meaningful here; clean_ly_subset stabilizes the joint angles
        OFF_STEPS=1000000
        ON_STEPS=1000000
        ON_ARGS=""
        ;;
    walker)
        ENV_ID=Walker2d-v5
        EQ_ARGS=""
        OFF_STEPS=1000000
        ON_STEPS=1000000
        ON_ARGS=""
        ;;
    *)
        echo "Unknown environment '$ENV' (use pendulum, double, quad, hopper or walker)"
        exit 1
        ;;
esac
