import torch
import argparse
    

class CHORDS:
    def __init__(
            self,
            T,
            x0,
            num_cores,
            solver=lambda x_t, score_t, t, s: x_t + score_t * (s - t),
            init_t=None,
            stopping_kwargs={
                "criterion": "core_index",
                "index": 1,
                "threshold": None,
            },
            verbose=False,
        ):
        self.T = T  # Number of steps
        # Each node (t, k) has three states
        # x[t, k] is empty,
        # x[t, k] is ready, but scores[t, k] is empty
        # Both x[t, k] and scores[t, k] are ready
        K = T + 1
        self.x_ready = torch.ones(T + 1, K) * -1
        self.scores_ready = torch.ones(T + 1, K) * -1
        self.num_cores = num_cores
        print('CHORDS using num_cores', self.num_cores)

        if init_t is None:
            raise NotImplementedError("Please provide initial points for CHORDS.")
        else:
            init_t = [int(t) for t in init_t.split('-')]
        assert len(init_t) == num_cores, f"Expected {num_cores} initial points, got {len(init_t)}"
        self.init_t = init_t

        print('CHORDS initial points', self.init_t)
        
        self.solver = solver
        self.counter = 0
        self.stopping_kwargs = stopping_kwargs
        print('CHORDS stopping criteria', self.stopping_kwargs)

        if stopping_kwargs["criterion"] not in ["core_index", "residual"]:
            raise ValueError("stopping_kwargs['criterion'] must be 'core_index' or 'residual'.")
        if stopping_kwargs["criterion"] == "core_index" and stopping_kwargs["index"] is None:
            raise ValueError("stopping_kwargs['index'] must be provided for core_index criterion.")
        if stopping_kwargs["criterion"] == "residual" and stopping_kwargs["threshold"] is None:
            raise ValueError("stopping_kwargs['threshold'] must be provided for residual criterion.")

        self.verbose = verbose

        self.x_ready[0, 0] = self.counter

        # maintains the starting point of each core
        self.cur_core_begin = [[0, 0, x0, None]]  # t, k, x, score
        # maintains the last point with score computed for each core
        self.cur_core_finish = [[None, None, None, None]]
        # maintains the current point for each core (w/o score)
        self.cur_core_status = [[0, 0, x0]]

        self.cur_core_to_compute = [0]  # this queue maintains the trajectories to compute

        # record the hits
        self.hits = []
        self.flops_count = 0

    def get_allocation(self):
        # Get the next batch of points to eval score
        cur_score_evals = []
        for core_id in self.cur_core_to_compute[:self.num_cores]:
            t, k, x = self.cur_core_status[core_id]
            assert self.scores_ready[t, k] == -1
            cur_score_evals.append((t, k, x))
        if self.verbose:
            print('*' * 10)
            print('cur_score_evals', [(t, k) for t, k, _ in cur_score_evals])
        return cur_score_evals
    
    def update_scores(self, scores):
        # Update scores and x from left to right
        for compute_id, (t, k, score) in enumerate(scores):
            # self.scores[t, k] = score
            core_id = self.cur_core_to_compute[compute_id]
            assert (self.cur_core_status[core_id][0], self.cur_core_status[core_id][1]) == (t, k)
            self.cur_core_finish[core_id] = (t, k, self.cur_core_status[core_id][2], score)
            if (self.cur_core_begin[core_id][0], self.cur_core_begin[core_id][1]) == (t, k):
                assert self.cur_core_begin[core_id][3] is None
                self.cur_core_begin[core_id][3] = score
            self.scores_ready[t, k] = self.counter + 1
            self.flops_count += 1

        self.counter += 1

    def schedule_cores(self):
        # deactivate the cores that have solved to self.T
        core_hit_idx = [core_id for core_id, (t, k, x) in enumerate(self.cur_core_status) if t == self.T]
        if len(core_hit_idx):
            print('core_hit_idx', core_hit_idx)
            for core_id in core_hit_idx:
                self.hits.append((self.cur_core_status[core_id][1], self.counter, self.cur_core_status[core_id][2]))
            self.cur_core_status = [core for core_id, core in enumerate(self.cur_core_status) if core_id not in core_hit_idx]
            self.cur_core_begin = [core for core_id, core in enumerate(self.cur_core_begin) if core_id not in core_hit_idx]
            self.cur_core_finish = [core for core_id, core in enumerate(self.cur_core_finish) if core_id not in core_hit_idx]
            
            # Early stopping criteria
            if self.stopping_kwargs["criterion"] == "core_index":
                if len(self.hits) - 1 >= self.stopping_kwargs["index"]:
                    print(f'CHORDS stopping since core {len(self.hits) - 1} terminates')
                    return core_hit_idx, True
            elif self.stopping_kwargs["criterion"] == "residual":
                # check if the L1 distance between the last two hits is small
                if len(self.hits) >= 2:
                    diff = torch.linalg.norm(self.hits[-1][-1] - self.hits[-2][-1]).double().item() / self.hits[-1][-1].numel()
                    if diff < self.stopping_kwargs["threshold"]:
                        print(f'CHORDS stopping since residual converged with diff {diff}')
                        return core_hit_idx, True

        if self.verbose:
            print('cur_core_begin', [(t, k) for t, k, x, score in self.cur_core_begin])
            # print('cur_core_finish',[(t, k) for t, k, x, score in self.cur_core_finish])
            print('cur_core_status', [(t, k) for t, k, x in self.cur_core_status])
            print('*' * 10)

        if len(self.cur_core_status) == 0:
            return [0], False
        
        return core_hit_idx, False
    
    def update_states(self, cnt):
        # update x from left to right
        for core_id in self.cur_core_to_compute[:cnt]:
            # decide whether can initialize new traj
            core_to_init = None
            if core_id == len(self.cur_core_status) - 1:
                # check whether is the k=0 point
                t_prev, k_prev, x_prev, score_prev = self.cur_core_finish[core_id]
                if k_prev == 0:
                    t_idx = self.init_t.index(t_prev)
                    if t_idx + 1 < len(self.init_t):
                        # initialize the next traj
                        t_next = self.init_t[t_idx + 1]
                        # coarse solve
                        x_next = self.solver(
                            x_prev, score_prev, t_prev, t_next,
                        )
                        core_to_init = core_id + 1
                        self.cur_core_status.append([t_next, 0, x_next])
                        self.cur_core_begin.append([t_next, 0, x_next, None])
                        self.cur_core_finish.append([None, None, None, None])
                    else:
                        # shoot to the end to get the "zero-th" hit
                        t_next = self.T
                        # coarse solve
                        x_next = self.solver(
                            x_prev, score_prev, t_prev, t_next,
                        )
                        self.hits.append((0, self.counter, x_next))

            # check whether current begin hits the previous finish at the same diffusion step
            hit_prev = core_id > 0 and self.cur_core_begin[core_id][0] == self.cur_core_finish[core_id - 1][0] and self.cur_core_begin[core_id][0] is not None
            if hit_prev:
                F = self.solver(
                    self.cur_core_finish[core_id][2],
                    self.cur_core_finish[core_id][3],
                    self.cur_core_finish[core_id][0],
                    self.cur_core_finish[core_id][0] + 1,
                )
                G = self.solver(
                    self.cur_core_begin[core_id][2],
                    self.cur_core_begin[core_id][3],
                    self.cur_core_begin[core_id][0],
                    self.cur_core_finish[core_id][0] + 1,
                )
                cur_G = self.solver(
                    self.cur_core_finish[core_id - 1][2],
                    self.cur_core_finish[core_id - 1][3],
                    self.cur_core_finish[core_id - 1][0],
                    self.cur_core_finish[core_id][0] + 1,
                )
                x = F - G + cur_G
                # update self
                self.cur_core_begin[core_id] = [self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, x, None]
                self.cur_core_status[core_id] = [self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, x]
            else:
                F = self.solver(
                    self.cur_core_finish[core_id][2],
                    self.cur_core_finish[core_id][3],
                    self.cur_core_finish[core_id][0],
                    self.cur_core_finish[core_id][0] + 1,
                )
                self.cur_core_status[core_id] = (self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, F)
            
            self.cur_core_to_compute.append(core_id)
            if core_to_init is not None:
                self.cur_core_to_compute.append(core_to_init)
    
    def get_last_x_and_hittime(self):
        hit_iter_idx = [hit[0] for hit in self.hits]
        hit_time = [hit[1] for hit in self.hits]
        hit_x = [hit[2] for hit in self.hits]
        return hit_iter_idx, hit_x, hit_time
    
    def get_flops_count(self):
        return self.flops_count


def test(num_cores, init_t):
    T = 50
    D = 128
    score_func = lambda x, tau: 0.1 * x + 0.01 * tau
    torch.manual_seed(10)
    x0 = torch.randn(D)
    # algo = CHORDS(T, x0, num_cores, init_t=init_t)

    algo = CHORDS(T, x0, num_cores, init_t=init_t, stopping_kwargs={
        "criterion": "residual",
        "index": 1,
        "threshold": 5,  # for convergence
    })

    while True:
        allocation = algo.get_allocation()
        if allocation  == []:
            break
        scores = []
        for t, k, x in allocation:
            scores.append((t, k, score_func(x, t)))
        algo.update_scores(scores)
        algo.update_states(len(allocation))
        delete_ids, earlystop = algo.schedule_cores()
        if earlystop:
            break
        
        algo.cur_core_to_compute = algo.cur_core_to_compute[len(allocation):]

        if len(delete_ids):
            algo.cur_core_to_compute = [core_id for core_id in algo.cur_core_to_compute if core_id not in delete_ids]
    
    hit_iter_idx, hit_x, hit_time = algo.get_last_x_and_hittime()
    print('init t', algo.init_t)
    print('hit iter', hit_iter_idx)
    print('hit time', [_ for _ in hit_time])
    converge_iter = None
    for itr, x in enumerate(hit_x):
        print(hit_time[itr], end=' ')
        print(x.sum())
        if (x.sum() - hit_x[-1].sum()).abs() < 1 and converge_iter is None:
            converge_iter = itr
    print('Time converge', hit_time[converge_iter])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cores', type=int, default=8)
    parser.add_argument('--init_t', type=str, default='0-2-4-8-16-24-32-40')
    args = parser.parse_args()
    test(args.num_cores, args.init_t)

