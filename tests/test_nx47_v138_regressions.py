import numpy as np

from nx47_vesu_kernel_v138 import V138Config, NX47EvolutionMemory, train_nx47_supervised, NX47V138Kernel


def test_v138_supervised_no_best_objective_keyerror():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(32, 6)).astype(np.float32)
    y_train = (rng.random(32) > 0.5).astype(np.float32)
    x_val = rng.normal(size=(16, 6)).astype(np.float32)
    y_val = (rng.random(16) > 0.5).astype(np.float32)
    cfg = V138Config(supervised_epochs=1, use_unet_25d=False)
    memory = NX47EvolutionMemory()

    _model, info = train_nx47_supervised(x_train, y_train, x_val, y_val, cfg, rng, memory)

    assert 'epoch_history' in info
    assert len(info['epoch_history']) >= 1


def test_v138_progress_bar_format():
    k = object.__new__(NX47V138Kernel)
    k.cfg = V138Config(progress_bar_width=10)
    bar = NX47V138Kernel._build_progress_bar(k, 50.0)
    assert bar.startswith('[') and bar.endswith(']')
    assert len(bar) == 12



def test_v138_plantracker_overall_progress_method(tmp_path):
    from nx47_vesu_kernel_v138 import PlanTracker

    tracker = PlanTracker(output_path=tmp_path / "roadmap.json")
    tracker.add_step("a", "A")
    tracker.add_step("b", "B")
    tracker.update("a", 20.0)
    tracker.update("b", 60.0)
    assert abs(tracker.overall_progress() - 40.0) < 1e-9
