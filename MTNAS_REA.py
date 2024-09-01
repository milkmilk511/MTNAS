
import pickle
import random
import copy
from nas_201_api import NASBench201API as API
from tqdm import tqdm
import math

_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

api = API('NAS-Bench-201-v1_1-096897.pth')


def get_spec_from_arch_str(arch_str):
    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]
    spec = [_opname_to_index[op] for node in nodes for op in node]
    return spec

idx_to_spec = {}
for i, arch_str in enumerate(api):
    idx_to_spec[i] = get_spec_from_arch_str(arch_str)

spec_to_idx = {}
for idx,spec in idx_to_spec.items():
    spec_to_idx[str(spec)] = idx




def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)

#randomly selected architecture sample from the search space
def random_spec():
    a = list(idx_to_spec.values())
    return random.choice(list(idx_to_spec.values()))
#The one-step mutation
def mutate_spec(old_spec,pop):
    count = 1000
    flag = True
    while (flag): #we hope to select a new architecture that has been found before.
        idx_to_change = random.randrange(len(old_spec))
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        new_entry = random.choice(possible_entries)
        new_spec = copy.copy(old_spec)
        new_spec[idx_to_change] = new_entry
        count = count - 1
        if new_spec not in pop:
            flag = False
        if count==0:
            flag = False
    return new_spec

def mutate_spec_tr(old_spec,syn):
    idx_to_change = random.randrange(len(old_spec))
    entry_to_change = old_spec[idx_to_change]
    possible_entries = [x for x in range(5) if x != entry_to_change]
    new_entry = random.choice(possible_entries)
    new_spec = copy.copy(old_spec)
    new_spec[idx_to_change] = new_entry
    new_est = syn[spec_to_idx[str(new_spec)]]
    old_est = syn[spec_to_idx[str(old_spec)]]
    if new_est>old_est:
        return new_spec
    else:
        return old_spec

#zero_cost value is applied to guide the one-step mutation
def mutate_spec_zero_cost(old_spec,old_est,synflow_proxy,pop,ran = None):
    n = 10
    candidates_space = [ind for ind in pop]
    if old_spec not in candidates_space:
        possible_specs = [(old_est,old_spec)]
    else:
        possible_specs = []
    for idx_to_change in range(len(old_spec)):
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry
            if new_spec not in candidates_space:
                possible_specs.append((synflow_proxy[spec_to_idx[str(new_spec)]], new_spec))
    if ran == True:#randomly select architectures from the generated individuals
        if len(possible_specs)<n:
            best_member = possible_specs
        else:
            best_member = []
            member = [i for i in range(len(possible_specs))]
            samples = random.sample(member, n)
            for i in samples:
                best_member.append(possible_specs[i])
    else:
        best_member = sorted(possible_specs, key=lambda i:i[0])# select architectures from the generated individuals based on their low-fidelity evaluation results.

    if len(best_member)<n:
        n = len(best_member)

    best_new_spec = []
    spec_l = len(best_member)
    for i in range(spec_l-n,spec_l):
        space = best_member[i][1]
        best_new_spec.append(space)
    return best_new_spec,n

def mutate_spec_zero_cost_or(old_spec,synflow_proxy,pop):
    candidates_space = [ind for ind in pop]
    possible_specs = []
    for idx_to_change in range(len(old_spec)):
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry
            if new_spec not in candidates_space:
                possible_specs.append((synflow_proxy[spec_to_idx[str(new_spec)]], new_spec))
    if len(possible_specs) == 0:
        return None
    best_new_spec = sorted(possible_specs, key=lambda i:i[0])[-1][1]
    return best_new_spec

#initialization of the REA NAS method
def initilize(dataset,pop_size = 10, warmup = 0,synflow_proxy=None):
    best_valids = [0.0]
    pop = []  # (validation, spec) tuples
    num_trained_models = 0
#if use the information from history tasks. Normally, it is False.
    if warmup > 0:
        zero_cost_pool = []
        for _ in range(warmup):
            spec = random_spec()
            spec_idx = spec_to_idx[str(spec)]
            zero_cost_pool.append((synflow_proxy[spec_idx], spec))
            zero_cost_pool = sorted(zero_cost_pool, key=lambda i: i[0], reverse=True)
    for i in range(pop_size):
        if warmup > 0:
            spec = zero_cost_pool[i][1]
        else:
            spec = random_spec()
        info = api.get_more_info(spec_to_idx[str(spec)], dataset, iepoch=None, hp='200', is_random=False)
        num_trained_models += 1
        pop.append((info['valid-accuracy'], spec))
        hist[dataset].append(spec)

        #update the search results. If the better architectures are found, there are added into the pool set.
        if info['valid-accuracy'] > best_valids[-1]:
            best_valids.append(info['valid-accuracy'])
            pool[dataset].append(spec)
        else:
            best_valids.append(best_valids[-1])
    best_valids.pop(0)
    return best_valids,pop

def run_evolution_search(valid,pop,turn,
                         tournament_size=5,
                         zero_cost_move=None, dataset = None,syn = None):

    reward = 0
    while (turn):
        sample = random_combination(pop, tournament_size) #parent selection
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        if zero_cost_move: #Apply zero_cost strategy to accelerate mutation
            new_spec = mutate_spec_zero_cost_or(best_spec,syn,hist[dataset])
        else:
            new_spec = mutate_spec(best_spec,hist[dataset])
        if new_spec == None:
            continue
        info = api.get_more_info(spec_to_idx[str(new_spec)], dataset, iepoch=None, hp='200', is_random=False) #architecture evaluation


        # kill the oldest individual in the population.
        pop.append((info['valid-accuracy'], new_spec))
        pop.pop(0)
        hist[dataset].append(new_spec)

        #update the architecture pool, if the better is found. Also, the reward is given for this action.
        if info['valid-accuracy'] > valid[-1]:
            valid.append(info['valid-accuracy'])
            pool[dataset].append(new_spec)
            reward = reward+1
        else:
            valid.append(valid[-1])

        turn = turn - 1
        if valid[-1]>task_oracle[dataset]:
            return reward,turn
    return reward,turn

#generate new architecture based on the promising architecture from other tasks
def transfer(dataset,pool,valid,pop,selected,zero = True):
    if dataset == 'cifar10-valid':
        synflow_proxy = synflow_proxy_c10
    elif dataset =='cifar100':
        synflow_proxy = synflow_proxy_c100
    else:
        synflow_proxy = synflow_proxy_img
    candidates = []
    best_est = 0
    best_space = None
    reward = 0
    for key in pool: #build the candadiate set of the promising architectures from other tasks
        if key != dataset:
            candidates = candidates+pool[key]
    for candidate in candidates:#based on the low-fidelity evaluation, i.e., zero_cost value, to select the promising architecture for transfer
        if candidate not in selected:
            spec_idx = spec_to_idx[str(candidate)]
            if synflow_proxy[spec_idx]>best_est:
                best_est = synflow_proxy[spec_idx]
                best_space = candidate
    if zero == True:
        if best_space == None: #if all the architectures in candidates has been reused, we randomly select one of them for transfer
            best_space = candidates[random.randint(0,len(candidates)-1)]
        selected.append(best_space)
        new_specs,n = mutate_spec_zero_cost(best_space,best_est,synflow_proxy,hist[dataset]) #zero_cost value is applied for architecture reuse
    elif zero == None and best_space != None:
        selected.append(best_space)
        new_specs = [best_space]
        n = 1
    else:
        return None,0

    number = 0
    for new_spec in new_specs:
        info = api.get_more_info(spec_to_idx[str(new_spec)], dataset, iepoch=None, hp='200', is_random=False)
        number = number+1
        # kill the oldest individual in the population.
        pop.append((info['valid-accuracy'], new_spec))
        pop.pop(0)
        hist[dataset].append(new_spec)
        if info['valid-accuracy'] > valid[-1]:
            valid.append(info['valid-accuracy'])
            reward = reward +1
            pool[dataset].append(new_spec)
        else:
            valid.append(valid[-1])

        if valid[-1]>task_oracle[dataset]:
            return reward,number

    return reward,n


#calculate the diversity of a population
def distance(pop):
    unit = 0.001

    pool_len = 10
    pool = sorted(pop, key=lambda i: i[0],reverse=True)
    dist = 0
    for i in range(pool_len-1):
        for j in range(i+1,pool_len):
            for k in range(6):
                if pool[i][1][k] != pool[j][1][k]:
                    dist = dist+unit
    return (dist-45*unit)


if __name__ =='__main__':
    #store the synflow value of each architecture on different task
    synflow_proxy_c10 = []
    synflow_proxy_c100 = []
    synflow_proxy_img = []

    #Refer paper "Zero-cost proxies for lightweight NAS" arXiv preprint arXiv:2101.08134 to calculate these values
    doc = ['cf10','cf100','im120']
    for name in doc:
        f = open('./nb2_'+name+'_seed42_dlrandom_dlinfo1_initwnone_initbnone.p', 'rb')
        index = []
        while (1):
            try:
                d = pickle.load(f)
                if d['i'] in index:
                    continue
                else:
                    index.append(d['i'])
                    if name == 'cf10':
                        synflow_proxy_c10.append(d['logmeasures']['synflow'])
                    elif name == 'cf100':
                        synflow_proxy_c100.append(d['logmeasures']['synflow'])
                    else:
                        synflow_proxy_img.append(d['logmeasures']['synflow'])
            except EOFError:
                break
        f.close()

    dataset1 = 'cifar10-valid' #cifar10, cifar100, ImageNet16-120
    dataset2 = "cifar100"
    dataset3 = "ImageNet16-120"

    #hyper-prameters
    alpha = 0.1

    #experiment settings
    num_rounds = 10
    turn = 10
    c10_valid = 0
    c100_valid = 0
    img_valid = 0

    #global solution of each task (e.g., the best validation accuracy)
    op_img = 46.7
    op_c100 = 73.49
    op_c10 = 91.60
    task_oracle = {'cifar10-valid': 91.60, 'cifar100': 73.49, "ImageNet16-120": 46.7}

    for _ in tqdm(range(num_rounds)):
        #random settings referring to "Zero-cost proxies for lightweight NAS" arXiv preprint arXiv:2101.08134
        random.seed(30+_)

        # initialization
        pool = {'cifar10-valid': [], 'cifar100': [], "ImageNet16-120": []}
        hist = {'cifar10-valid': [], 'cifar100': [], "ImageNet16-120": []}
        c10_best_valids, pop_10 = initilize(dataset=dataset1)
        c100_best_valids, pop_100 = initilize(dataset=dataset2)
        img_best_valids, pop_img = initilize(dataset=dataset3)

        selected1 = []
        selected2 = []
        selected3 = []
        reward_c10_sl, reward_c10_tr, q_c10, p_c10 = 0, 0, 0, 0
        reward_c100_sl, reward_c100_tr, q_c100, p_c100 = 0, 0, 0, 0
        reward_img_sl, reward_img_tr, q_img, p_img = 0, 0, 0, 0

        total_10 = 10
        total_100 = 10
        total_img = 10
        #We solve the problems in a sequential way.
        while (c10_best_valids[-1] < op_c10 or c100_best_valids[-1] < op_c100 or img_best_valids[-1] < op_img):
            if c10_best_valids[-1] < op_c10:
                #Architecture search on CIFAR-10
                q_c10 = alpha * q_c10 +  reward_c10_sl
                p_c10 = alpha * p_c10 +  reward_c10_tr
                dist_c10 = distance(pop_10)
                prob_c10 = (1+math.exp(-1*(dist_c10+q_c10-p_c10)))**(-1) #tranfer probability update
                r_10 = random.random()
                reward_c10_sl = 0
                reward_c10_tr = 0

                if r_10<prob_c10:#self-evaluation
                    reward_c10_sl,num = run_evolution_search(c10_best_valids, pop_10, turn, dataset=dataset1, syn=synflow_proxy_c10)
                    reward_c100_tr = reward_c100_tr +reward_c10_sl
                    reward_img_tr = reward_img_tr + reward_c10_sl
                    total_10 = total_10+(turn-num)
                else:#knowledge transfer
                    reward_tr,n = transfer(dataset1, pool,c10_best_valids,pop_10,selected1)
                    if reward_tr == None:
                        pass
                    else:
                        reward_c10_sl = reward_c10_sl + reward_tr
                        reward_c100_tr = reward_c100_tr + reward_tr
                        reward_img_tr = reward_img_tr + reward_tr
                        total_10 =total_10+n

            if c100_best_valids[-1] < op_c100:
                #Architecture search on CIFAR-100
                q_c100 = alpha * q_c100 + reward_c100_sl
                p_c100 = alpha * p_c100 + reward_c100_tr
                dist_c100 = distance(pop_100)
                prob_c100 = (1 + math.exp(-1*(dist_c100+q_c100-p_c100))) ** (-1)
                r_100 = random.random()
                reward_c100_sl = 0
                reward_c100_tr = 0
                if r_100 < prob_c100:
                    reward_c100_sl,num = run_evolution_search(c100_best_valids, pop_100, turn, dataset=dataset2,syn = synflow_proxy_c100)
                    reward_c10_tr = reward_c10_tr + reward_c100_sl
                    reward_img_tr = reward_img_tr + reward_c100_sl
                    total_100 = total_100+(turn-num)
                else:
                    reward_tr,n = transfer(dataset2, pool,c100_best_valids, pop_100,selected2)
                    if reward_tr == None:
                        pass
                    else:
                        reward_c10_tr = reward_c10_tr + reward_tr
                        reward_c100_sl= reward_c100_sl + reward_tr
                        reward_img_tr = reward_img_tr + reward_tr
                        total_100 = total_100+n

            if img_best_valids[-1] < op_img:
                #img
                q_img = alpha * q_img + reward_img_sl
                p_img = alpha * p_img + reward_img_tr
                dist_img = distance(pop_img)
                prob_img = (1 + math.exp(-1*(dist_img+q_img-p_img))) ** (-1)
                r_img = random.random()
                reward_img_sl = 0
                reward_img_tr = 0


                if r_img < prob_img:
                    reward_img_sl,num = run_evolution_search(img_best_valids, pop_img, turn, dataset=dataset3,syn = synflow_proxy_img)
                    reward_c10_tr = reward_c10_tr + reward_img_sl
                    reward_c100_tr = reward_c100_tr + reward_img_sl
                    total_img = total_img+(turn-num)
                else:
                    reward_tr,n = transfer(dataset3, pool, img_best_valids, pop_img,selected3)
                    if reward_tr == None:
                        pass
                    else:
                        reward_c10_tr = reward_c10_tr + reward_tr
                        reward_c100_tr = reward_c100_tr + reward_tr
                        reward_img_sl = reward_img_sl + reward_tr
                        total_img = total_img+n


        c10_valid =c10_valid+ total_10
        c100_valid = c100_valid + total_100
        img_valid = img_valid + total_img



    c10_valid = c10_valid/num_rounds
    c100_valid = c100_valid/num_rounds
    img_valid = img_valid/ num_rounds


    print('c10_ave:',c10_valid ,'c100_ave:',c100_valid,'img_ave:',img_valid )
    print('total:', c10_valid+c100_valid+img_valid)