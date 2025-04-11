import sys
import itertools
from collections import deque
from math import log, exp
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import scipy.special
import pandas as pd
import tqdm

import utils
import pmonad

def demonstrate_systematic_learning_22(num_samples=1000, num_runs=1, **kwds):
    """ 2^2!=24 possible languages. """
    # works with redundancy=2, positional=False, split_alpha=1
    # final posterior is 1/2 because of symmetry
    targets = [0, 1, 7]
    df, support, probs = learn_from_samples(K=2, V=2, num_samples=num_samples, targets=targets*num_runs, **kwds)
    grammars = list(itertools.permutations([x+'#' for x in support]))
    def gen():
        for target in targets:
            curves = il.curves_from_sequences(grammars[target], np.exp(probs))
            yield {
                'target': target,
                'grammar': grammars[target],
                'ee': il.ee(curves),
                'ms_auc': il.ms_auc(curves),
            }
    return df, probs, pd.DataFrame(gen())    
    

def demonstrate_systematic_learning_23(num_samples=20000, num_runs=1, **kwds):
    """ 2**3!=40320 possible languages. """
    targets = [
        0, # id(1,2,3)
        1, # toffoli(1,2,3) -- flip on more frequent control bits
        5040, # toffoli(not(1), not(2), 3) -- flip on less frequent control bits
        121, # cnot(2,3) -- flip on more frequent control bit
        5046, # cnot(not(2), 3) -- flip on less frequent control bit
        11536, # weakly systematic -- if positional=True, no different from strongly systematic
        10000, # nonsystematic
        15000,
        20000,
        25000,
        30000, # nonsystematic
    ]
    df, support, probs = learn_from_samples(
        K=3,
        V=2,
        num_samples=num_samples,
        targets=tqdm.tqdm(targets*num_runs),
        **kwds
    )
    grammars = list(itertools.permutations([x+'#' for x in support]))    
    def gen():
        for target in targets:
            curves = il.curves_from_sequences(grammars[target], np.exp(probs))
            yield {
                'target': target,
                'grammar': grammars[target],
                'ee': il.ee(curves),
                'ms_auc': il.ms_auc(curves),
            }
    return df, probs, pd.DataFrame(gen())

def learn_from_samples(
        targets=[0],
        K=3, # sequence length
        V=2, # vocabulary size
        p=.6,
        asymmetry=.1,
        truncate=True,
        split_alpha=.5, # sample substrings of size k with probability \propto split_alpha^k. 
        maxlen=None,
        num_samples=10000,
        redundancy=1,
        parallel=True,
        full_support=False,
        positional=False):

    """
    Language learning as cryptography. The grammar is the key to be deciphered.
    Given uniformly distribution s = f_K(m), and knowledge of p(m),
    and lots of samples of s, how long to determine K?

    Intuitively if the data distribution is flat, then the key is harder to recover,
    because there are more keys that give rise to that distribution. On the other hand,
    if the data distribution is peaky, then it is easier to recover the key,z
    because structure-preserving keys are rare. In cryptography, a good code has
    "diffusion" and "confusion", which serve to create a flat data distribution. For
    learnability, contrariwise, you want to MINIMIZE diffusion and confusion.

    If the samples to be learned from are truncated---consisting of random contiguous substrings
    from utterances---then high E should be detrimental to learning, because long-range
    correlations will be missed. Therefore, a language is more "learnable" with small samples
    if it has low E.
    """
    if maxlen is None:
        maxlen = K
    alphabet = [chr(65+v) for v in range(V)]
    if V == 2:
        probs = np.array([p, 1-p])
    else:
        raise NotImplementedError
    character_sources = [
        pmonad.Enumeration(list(zip(alphabet, np.log(probs+k*asymmetry*np.array([+1,-1])))))
        for k in range(K)
    ]
    string_source = character_sources[0].ret("")
    
    @string_source.lift_ret
    def concatenate_distinctly(c, x):
        if positional:
            return "".join([c, chr(len(c)*V + ord(x))*redundancy])
        else:
            return "".join([c, x*redundancy])
        
    for k in range(K):
        string_source = string_source >> (lambda c:
            character_sources[k] >> (lambda x:
            concatenate_distinctly(c, x)))

    source_probs = [p for _, p in string_source]
    source = pmonad.Enumeration(list(enumerate(source_probs)))
    
    if full_support:
        assert not positional
        strings = list(utils.cartesian_forms(V, K*redundancy))
    else:
        strings = [s for s, _ in string_source]

    grammars_support = list(itertools.permutations(strings, len(source_probs)))
    num_grammars = len(grammars_support)

    @source.lift_ret
    def produce(g, m):
        return grammars_support[g][m]

    if truncate:
        obs_support = list({
            obs
            for s in strings
            for k in range(maxlen)
            for obs in utils.padded_sliding(s, k+1)
        })
    else:
        obs_support = strings
    num_obs = len(obs_support)

    context_size = source.exponential(range(maxlen), alpha=-np.log(split_alpha))

    mul, from_p, to_log, zero = source.field.mul, source.field.from_p, source.field.to_log, source.field.zero    
    def likelihood(g):
        # m ~ source
        # utt ~ produce(g, m)
        utterances = source >> (lambda m: produce(g, m))
        if truncate:
            # adjust probabilities
            adjusted = type(source)(
                (x, mul(from_p(len(x)+1), p))
                for x, p in utterances.values
            )
            # utt ~ adjusted
            # k ~ context_size
            # result ~ uniform(sliding(utt, k))
            return adjusted >> (lambda utt:
                context_size >> (lambda k:
                source.uniform(utils.padded_sliding(utt, k+1))))
        else:
            return utterances

    # arrange into joint probabilities of shape GO, evidence of shape O
    # problem: O has different support size if we are using full support
    print("Calculating likelihood array...", file=sys.stderr)
    likelihood_dicts = map(dict, map(likelihood, range(num_grammars)))
    likelihood_array = np.array([
        [
            to_log(likelihood_dict.get(obs, zero))
            for obs in obs_support
        ]
        for likelihood_dict in tqdm.tqdm(likelihood_dicts, total=num_grammars)
    ])

    def run_target(target):
        #print("Running target grammar %d" % target, file=sys.stderr)
        def gen():
            target_likelihood = np.exp(likelihood_array[target])
            log_prior_array = -log(num_grammars) * np.ones(num_grammars)
            for i in range(num_samples):
                observed = np.random.choice(range(num_obs), p=target_likelihood)
                joint = log_prior_array + likelihood_array[:, observed] # shape G
                log_posterior_array = scipy.special.log_softmax(joint, -1) # shape G
                #marginal = log_prior_array[:, None] + likelihood_array
                yield {
                    'target': str(target),
                    'i': i,
                    'data': obs_support[observed],
                    'posterior': log_posterior_array[target],
                    'entropy': scipy.stats.entropy(np.exp(log_posterior_array)),
                    #'formal_entropy': ...,
                    #'semantic_entropy': ...,
                }
                log_prior_array = log_posterior_array
        return list(gen())

    print("Simulating learning...", file=sys.stderr)
    if parallel:
        with Pool() as p:
            results = tqdm.tqdm(p.map(run_target, targets))
    else:
        results = tqdm.tqdm(map(run_target, targets))
        
    return pd.DataFrame(utils.flat(results)), strings, source_probs
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Simulate learning of very simple languages.")
    parser.add_argument("config", type=str, help="'noiseless', 'noisy', or 'redundancy'")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples.")
    parser.add_argument("--num_runs", type=int, default=10000, help="Number of runs.")
    parser.add_argument("--bias", type=float, default=0.79, help="Coinflip bias value.")
    parser.add_argument("--noise_level", type=float, default=0.5, help="Erasure rate.")

    args = parser.parse_args()
    
    if args.config == "noiseless":
        df, strings, source = learn_from_samples(
            K=3,
            V=2,
            num_samples=args.num_samples,
            truncate=False,
            p=args.bias,
            targets=[0]*args.num_runs,
        )
    elif args.config == "noisy":
        df, strings, source = learn_from_samples(
            K=3,
            V=2,
            num_samples=args.num_samples,
            truncate=True,
            p=args.bias,
            split_alpha=1-args.noise_level,
            targets=[0,37965,10000,20000]*args.num_runs,
        )        
    elif args.config == "redundancy":
        df, strings, source = learn_from_samples(
            K=2,
            V=2,
            num_samples=args.num_samples,
            truncate=True,
            redundancy=2,
            p=args.bias,
            split_alpha=1-args.noise_level,
            targets=[0,844]*args.num_runs,
            full_support=True,
        )                

    df.to_csv(sys.stdout)
        
