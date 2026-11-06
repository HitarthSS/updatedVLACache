"""Preflight -> separate development -> freeze winner -> matched evaluation."""
import argparse
import fcntl
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent


def array_sha(a):
    import numpy as np
    a = np.asarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def source_manifest():
    from runtime import sha, SOURCE, GUARDED_SOURCE
    paths = ['runtime.py', 'my_harness.py', 'experiment.py', 'rendering.py', 'PROTOCOL.md', 'launch.py']
    repos = {}
    for p in ['upstream', 'transformers', 'LIBERO']:
        repos[p] = subprocess.check_output(['git', '-C', str(ROOT / p), 'rev-parse', 'HEAD'], text=True).strip()
    return {'source_hashes': {p: sha(ROOT / p) for p in paths}, 'repositories': repos,
            'native_action_function_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(),
            'common_guarded_function_sha256': hashlib.sha256(GUARDED_SOURCE.encode()).hexdigest(),
            'model': json.loads((ROOT / 'MODEL_READY.json').read_text()),
            'processor': json.loads((ROOT / 'PROCESSOR_READY.json').read_text()),
            'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'protocol': {'suite': 'libero_spatial', 'dev_init_indices': [40],
                'final_init_indices': list(range(20, 32)), 'environment_seed': 0, 'model_seed': 7,
                'control_horizon': 220, 'settling_actions': 10, 'maximum_physical_actions': 230,
                'final_arms': ['bare', 'authors', 'mine'], 'candidates': ['event_refresh', 'anchor_ttl', 'stall_refresh'],
                'paid_api_calls': 0, 'external_tokens': 0}}


def observations(env, obs):
    from runtime import check_camera, libero_utils
    import numpy as np
    health = check_camera(obs)
    image = libero_utils.get_libero_image(obs, 224)
    return image, np.asarray(obs['robot0_eef_pos']).copy(), np.asarray(obs['robot0_gripper_qpos']).copy(), health


def create_env(task, init):
    from runtime import robot_utils, libero_utils, check_camera
    robot_utils.set_seed_everywhere(7)
    env, text = libero_utils.get_libero_env(task, 'openvla', resolution=256)
    env.reset()
    obs = env.set_init_state(init)
    check_camera(obs)
    setup_seconds = 0.
    for _ in range(10):
        t = time.monotonic()
        obs, reward, done, info = env.step(libero_utils.get_libero_dummy_action('openvla'))
        setup_seconds += time.monotonic() - t
        check_camera(obs)
        if done:
            raise RuntimeError('Task already successful during stabilization; no case replacement allowed')
    state = env.sim.get_state().flatten().copy()
    image, position, fingers, health = observations(env, obs)
    initial = {'source_initial_state_sha256': array_sha(init), 'post_settle_state_sha256': array_sha(state),
               'post_settle_rgb_sha256': array_sha(image), 'position_sha256': array_sha(position),
               'fingers_sha256': array_sha(fingers)}
    return env, obs, text, initial, setup_seconds


def close(env):
    env.close()
    del env
    gc.collect()


def preflight(policy, suite):
    from runtime import atomic, robot_utils, torch, ORIGINAL_ACTION, GUARDED_ACTION, sha
    import numpy as np
    from PIL import Image
    from my_harness import Controller
    # A six-frame public-observation tape from a separate official initial state.
    env, obs, text, initial, _ = create_env(suite.get_task(0), suite.get_task_init_states(0)[49])
    tape = []
    try:
        for i in range(6):
            image, pos, fingers, _ = observations(env, obs)
            tape.append((image, pos, fingers))
            obs, _, _, _ = env.step([0., 0., .01 if i < 3 else -.01, 0., 0., 0., -1.])
    finally:
        close(env)
    parity = []
    for arm in ['bare', 'authors']:
        policy.start(arm)
        old = [policy.act(*o, text, original=True) for o in tape]
        policy.start(arm)
        new = [policy.act(*o, text) for o in tape]
        for i, ((a, x), (b, y)) in enumerate(zip(old, new)):
            assert np.array_equal(a, b), (arm, i, a, b)
            assert x['prefill_token_layers'] == y['prefill_token_layers']
            parity.append({'arm': arm, 'frame': i, 'bitwise_equal': True,
                           'reusable_tokens': y['reusable_tokens']})
    assert any(p['reusable_tokens'] > 0 for p in parity), 'Cache was never exercised'
    # Force the empty-selection edge case; its result must equal a fresh native action.
    import runtime
    saved = runtime.ORIGINAL_SELECTION
    try:
        policy.start('authors')
        policy.act(*tape[0], text)
        runtime.ORIGINAL_SELECTION = lambda attn, image, stable: (np.asarray(image), [])
        fallback, stats = policy.act(*tape[1], text)
        assert stats['empty_mask_full_refresh']
        policy.start('bare')
        exact, _ = policy.act(*tape[1], text)
        assert np.array_equal(fallback, exact), 'Empty-mask fallback changed native full inference'
    finally:
        runtime.ORIGINAL_SELECTION = saved
    # Candidate first predictions are identical; no stale cross-episode memory.
    policy.start('authors')
    reference, _ = policy.act(*tape[0], text)
    for name in ['event_refresh', 'anchor_ttl', 'stall_refresh']:
        policy.start(name)
        action, _ = policy.act(*tape[0], text)
        assert np.array_equal(action, reference), name
    # The patch-age bound and scene-anchor test must prevent indefinite stale reuse.
    c = Controller('anchor_ttl')
    im = np.full((224, 224, 3), 100, dtype=np.uint8)
    c.after(np.zeros(7), im, False)
    for i in range(7):
        allowed = c.filter_indices(im, [1, 2])
        if i == 6:
            assert allowed == [], 'Six consecutive potential reuses must force refresh'
        c.after(np.zeros(7), im, bool(allowed))
    changed = im.copy(); changed[:14, :14, :] = [255, 0, 0]
    assert 1 not in c.filter_indices(changed, [1, 2])
    atomic(ROOT / 'PREFLIGHT.json', {'status': 'PASS', 'parity': parity,
          'empty_mask_fallback_matches_bare_bitwise': True, 'candidate_first_action_parity': True,
          'token_age_and_anchor_checks': True, 'initial_state': initial,
          'smoke_init_index': 49, 'smoke_actions': 6, 'smoke_excluded_from_evaluation': True,
          'policy_parameter_hash': policy.weights_sha(), 'source_manifest': source_manifest(), 'paid_api_calls': 0})


def episode(policy, suite, task_id, init_index, arm, method, phase, expected=None, max_steps=220):
    from runtime import atomic, robot_utils, check_camera
    import numpy as np
    from PIL import Image
    task = suite.get_task(task_id)
    case = f'task-{task_id:02d}-init-{init_index:02d}'
    out = ROOT / phase / arm / case
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing episode: ' + str(out))
    out.mkdir(parents=True)
    start = time.monotonic()
    atomic(ROOT / 'status.json', {'state': 'running', 'phase': phase, 'case': case,
                                 'arm': arm, 'method': method, 'time': time.time(), 'pid': os.getpid()})
    env, obs, text, initial, simulator_seconds = create_env(task, suite.get_task_init_states(task_id)[init_index])
    try:
        if expected is not None:
            assert initial == expected, ('Paired initial mismatch BEFORE action', case, arm, initial, expected)
        atomic(out / 'INITIAL.json', initial)
        Image.fromarray(observations(env, obs)[0]).save(out / 'initial.png')
        policy.start(method)
        counters = {k: 0. for k in ['policy_seconds', 'full_policy_wall_seconds', 'prefill_token_layers',
                      'decode_token_layers', 'decoder_flops', 'decoder_cuda_ms', 'input_tokens',
                      'reusable_tokens', 'empty_mask_full_refresh']}
        success = False
        trace = (out / 'decisions.jsonl').open('w', buffering=1)
        first_action_sha = None
        rollout_start = time.monotonic()
        for step in range(max_steps):
            image, position, fingers, health = observations(env, obs)
            t = time.monotonic()
            action, stats = policy.act(image, position, fingers, text)
            counters['full_policy_wall_seconds'] += time.monotonic() - t
            if first_action_sha is None:
                first_action_sha = array_sha(action)
            for key in counters:
                if key != 'full_policy_wall_seconds':
                    counters[key] += float(stats.get(key, 0.))
            executed = robot_utils.invert_gripper_action(robot_utils.normalize_gripper_action(action.copy(), binarize=True))
            t = time.monotonic()
            obs, reward, done, info = env.step(executed.tolist())
            simulator_seconds += time.monotonic() - t
            check_camera(obs)
            record = {'step': step + 1, 'raw_action': action.tolist(), 'executed_action': executed.tolist(),
                      'public_eef_position': position.tolist(), 'public_fingers': fingers.tolist(),
                      'camera_health': health, **stats}
            trace.write(json.dumps(record) + '\n')
            if step % 16 == 0:
                atomic(ROOT / 'live.json', {'phase': phase, 'case': case, 'arm': arm, 'method': method,
                            'step': step + 1, 'horizon': max_steps, 'time': time.time()})
            if step % 64 == 0:
                Image.fromarray(image).save(out / f'frame-{step:03d}.png')
            if done:
                success = True
                break
        trace.close()
        control_steps = step + 1
        assert control_steps <= 220 and control_steps + 10 <= 230
        result = {'case': case, 'task_id': task_id, 'task': task.name, 'instruction': text,
                  'init_index': init_index, 'arm': arm, 'method': method, 'phase': phase,
                  'initial': initial, 'first_action_sha256': first_action_sha,
                  'success': success, 'control_steps': control_steps, 'settling_steps': 10,
                  'physical_steps': control_steps + 10, 'control_horizon': 220,
                  'horizon_violations': 0, 'policy_calls': control_steps, 'output_action_tokens': 7 * control_steps,
                  'external_tokens': 0, 'paid_api_calls': 0,
                  'simulator_seconds': simulator_seconds, 'wall_seconds': time.monotonic() - start,
                  'rollout_seconds': time.monotonic() - rollout_start, **counters}
        atomic(out / 'RESULT.json', result)
        return result
    finally:
        close(env)


def summarize(records, arms):
    import numpy as np
    result = {}
    cases = sorted(set(r['case'] for r in records))
    by_case = {case: {r['arm']: r for r in records if r['case'] == case} for case in cases}
    matched = [rows for rows in by_case.values() if all(arm in rows for arm in arms)]
    for arm in arms:
        rows = [pair[arm] for pair in matched]
        result[arm] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows)}
        result[arm]['success_percent'] = 100 * result[arm]['successes'] / max(len(rows), 1)
        for key in ['policy_calls', 'input_tokens', 'output_action_tokens', 'physical_steps', 'control_steps',
                    'prefill_token_layers', 'decode_token_layers', 'decoder_flops', 'decoder_cuda_ms',
                    'policy_seconds', 'full_policy_wall_seconds', 'wall_seconds', 'simulator_seconds',
                    'horizon_violations', 'paid_api_calls', 'external_tokens', 'empty_mask_full_refresh']:
            result[arm][key] = sum(r[key] for r in rows)
    result['completed_matched_cases'] = len(matched)
    result['completed_episodes'] = len(records)
    result['matched_initials'] = all(all(rows[a]['initial'] == rows[arms[0]]['initial'] for a in arms) for rows in matched)
    if 'mine' in arms and matched:
        differences = np.array([r['mine']['success'] - r['authors']['success'] for r in matched], dtype=float)
        rng = np.random.default_rng(20261101)
        draws = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1) * 100
        helped = sum(r['mine']['success'] and not r['authors']['success'] for r in matched)
        harmed = sum(r['authors']['success'] and not r['mine']['success'] for r in matched)
        discordant = helped + harmed
        p = 1. if not discordant else min(1., 2 * sum(math.comb(discordant, i) for i in range(min(helped, harmed) + 1)) / 2**discordant)
        per_task = [[r['mine']['success'] - r['authors']['success'] for r in matched
                     if r['authors']['task_id'] == t] for t in range(10)]
        means = np.array([np.mean(x) for x in per_task if x])
        cluster = rng.choice(means, (10000, len(means)), replace=True).mean(axis=1) * 100
        result['paired'] = {'helped': helped, 'harmed': harmed, 'difference_pp': float(differences.mean() * 100),
              'episode_bootstrap_95_pp': np.percentile(draws, [2.5, 97.5]).tolist(),
              'task_cluster_bootstrap_95_pp': np.percentile(cluster, [2.5, 97.5]).tolist(), 'mcnemar_exact_p': p}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from runtime import Policy, atomic, install_renderer, sha
    from libero.libero import benchmark
    import numpy as np
    install_renderer()
    suite = benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    assert suite.n_tasks == 10
    assets = []
    for task_id in range(10):
        starts = suite.get_task_init_states(task_id)
        assert len(starts) >= 50
        assets.append({'task_id': task_id, 'task': suite.get_task(task_id).name,
                       'start_count': len(starts), 'state_bank_sha256': array_sha(starts.numpy() if hasattr(starts, 'numpy') else starts)})
    atomic(ROOT / 'EVALUATION_ASSETS.json', assets)
    policy = Policy()
    initial_weights = policy.weights_sha()
    preflight(policy, suite)
    if args.preflight_only:
        atomic(ROOT / 'status.json', {'state': 'preflight_passed', 'time': time.time()})
        return
    manifest = source_manifest()
    manifest.update(parameter_sha256=initial_weights, assets=assets, frozen_unix=time.time())
    atomic(ROOT / 'DEV_FROZEN.json', manifest)
    methods = ['bare', 'authors', 'event_refresh', 'anchor_ttl', 'stall_refresh']
    dev = []
    for task_id in range(10):
        expected = None
        order = methods[task_id % 5:] + methods[:task_id % 5]
        for arm in order:
            row = episode(policy, suite, task_id, 40, arm, arm, 'development', expected)
            expected = row['initial'] if expected is None else expected
            dev.append(row)
            atomic(ROOT / 'DEVELOPMENT_SUMMARY.json', summarize(dev, methods))
    assert policy.weights_sha() == initial_weights
    summary = summarize(dev, methods)
    # Fixed before dev: maximize successes, then minimize deterministic decoder
    # token-layer work, then a fixed alphabetical tie break. Timing never selects.
    choices = ['event_refresh', 'anchor_ttl', 'stall_refresh']
    winner = min(choices, key=lambda name: (-summary[name]['successes'],
                          summary[name]['prefill_token_layers'] + summary[name]['decode_token_layers'], name))
    selection = {'winner': winner, 'ranking': sorted(choices, key=lambda name: (-summary[name]['successes'],
                  summary[name]['prefill_token_layers'] + summary[name]['decode_token_layers'], name)),
                 'selection_rule': 'success count, then total decoder token-layer work, then alphabetical',
                 'summary': summary, 'development_cases': 10, 'development_episodes': 50,
                 'development_gain_vs_authors_pp': summary[winner]['success_percent'] - summary['authors']['success_percent'],
                 'no_improvement_claim_from_development': True, 'time': time.time()}
    atomic(ROOT / 'SELECTION.json', selection)
    assert source_manifest()['source_hashes'] == manifest['source_hashes']
    final_frozen = dict(manifest, selected_method=winner, frozen_unix=time.time(),
                        selection_sha256=sha(ROOT / 'SELECTION.json'), preregistered_cases=120, physical_episodes=360)
    atomic(ROOT / 'FROZEN.json', final_frozen)
    final = []
    arms = ['bare', 'authors', 'mine']
    orders = list(itertools.permutations(arms))
    for state_offset in range(12):
        for task_id in range(10):
            case_idx = state_offset * 10 + task_id
            expected = None
            for arm in orders[case_idx % len(orders)]:
                row = episode(policy, suite, task_id, 20 + state_offset, arm,
                              winner if arm == 'mine' else arm, 'evaluation', expected)
                expected = row['initial'] if expected is None else expected
                final.append(row)
                atomic(ROOT / 'SUMMARY.json', summarize(final, arms))
    result = summarize(final, arms)
    assert result['completed_matched_cases'] == 120 and result['completed_episodes'] == 360
    assert result['matched_initials']
    assert policy.weights_sha() == initial_weights
    assert source_manifest()['source_hashes'] == manifest['source_hashes']
    result['paper_claims_spatial'] = {'bare_policy_percent': 84.4, 'authors_cache_percent': 83.8}
    result['selected_method'] = winner
    result['evaluation_scope'] = '10 reported LIBERO-Spatial tasks x 12 released initial states, smaller than the 50/task default'
    result['full_paper_reproduction'] = False
    result['shared_gpu_timing_is_descriptive'] = True
    atomic(ROOT / 'SUMMARY.json', result)
    atomic(ROOT / 'AUDIT.json', {'status': 'PASS', 'complete': True, 'matched_cases': 120,
             'physical_episodes': 360, 'weights_unchanged': True, 'sources_unchanged': True,
             'horizon_violations': 0, 'paired_initials_match': True, 'paid_api_calls': 0})
    atomic(ROOT / 'status.json', {'state': 'complete', 'completed_cases': 120, 'completed_episodes': 360,
                                 'winner': winner, 'finished_unix': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        p = ROOT / 'LAST_ERROR.json'
        p.write_text(json.dumps({'time': time.time(), 'pid': os.getpid(), 'traceback': traceback.format_exc()}, indent=2))
        raise
