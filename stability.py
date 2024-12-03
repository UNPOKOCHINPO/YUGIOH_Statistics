import random
def read_file_to_2d_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    result = []
    for line in lines:
        elements = line.strip().split(',')
        processed_elements = []
        for element in elements:
            if element.isdigit():
                processed_elements.append(int(element))
            else:
                try:
                    processed_elements.append(float(element))
                except ValueError:
                    processed_elements.append(element)
        result.append(processed_elements)
    
    return result

def calculate_score(hands,idx=1):
    score = 0
    while len(hands)>0:
        card = hands[0]
        single_score = card[idx]
        counter = sum([each_card.count(card[0]) for each_card in hands])
        if 'noTurn1' in card:
            score += counter * single_score
        elif counter==1:
            score += single_score
        elif counter==2:
            score += single_score * (1+1/2)
        else:
            score += single_score * (1+1/2+1/4)
        hands = [c for c in hands if card[0] not in c]
    return score

def calculate_average_score_with_std(file_name,trial_number=10**6,first=1/2,second=1/2):
    random.seed(123)
    deck = read_file_to_2d_list(file_name)
    scores = []
    for _ in range(trial_number):
        hands1 = random.sample(deck,5)
        hands2 = random.sample(deck,6)

        score1 = calculate_score(hands1,1)
        score1 = score1 / 5
        
        score2 = 0
        draw = hands2[0]
        hands2 = hands2[1:]

        score2 = calculate_score(hands2,2)
        count_draw = sum([hand.count(draw[0]) for hand in hands2])
        # 後攻1ターン目の誘発はターン１あろうがなかろうが結構ごみ
        # 特に、重ね引いたらごみ
        if 'yuhatsu' in draw:
            single_score = draw[2]/2**(1+count_draw)
        else:
            single_score = draw[2]/2**(count_draw)
        score2 += single_score
        score2 = score2 / 6

        
        score = first*score1 + second*score2
        scores.append(score)
    return scores

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def plot_hist(scores,file_name,start=3,end=8):
    width = 0.2
    # 度数分布を計算
    bins = np.arange(start, end+width, width)
    hist, bin_edges = np.histogram(scores, bins=bins)
    hist = hist/len(scores)
    print(np.sum(hist))
    # 度数分布表を表示
    print("度数分布表:")
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {hist[i]}")

    # 度数分布をグラフで表示
    #plt.bar(bin_edges[:-1], hist, width=width, edgecolor='black')
    plt.hist(scores, bins=bin_edges[:-1], weights=np.ones_like(scores) / len(scores), density=True, alpha=0.6, edgecolor='black')
    

    mean, std = norm.fit(scores)
    """xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    pdf = norm.pdf(x, mean, std)
    
    plt.plot(x, pdf, 'k-', linewidth=2)"""
    
    plt.xlabel(r'Hand Score $S$')
    plt.ylabel('Probability')
    plt.xticks(np.arange(start, end+1, 1))

    # 平均値と標準偏差の線を追加
    # 平均値と標準偏差を計算
    print(mean)
    print(std)
    plt.axvline(mean, color='r', linewidth=1, label=r'$\overline{S}$')
    plt.axvline(mean + std, color='g', linewidth=1, label='$\overline{S}\pm\sigma$')
    plt.axvline(mean - std, color='g', linewidth=1)
    plt.axvline(mean + 2 * std, color='b', linewidth=1, label='$\overline{S}\pm 2\sigma$')
    plt.axvline(mean - 2 * std, color='b', linewidth=1)

    plt.legend(loc='upper left')
    plt.savefig(file_name[:-4]+'.png')
    plt.close('all')



if __name__=='__main__':
    trial_number=10**6
    first = 0.05
    file_name = 'mikanko.txt'
    scores_with_ex_second = calculate_average_score_with_std(file_name=file_name,trial_number=trial_number,first=first,second=1-first)
    plot_hist(scores_with_ex_second,file_name,end=10)
    file_name = 'mikanko_makuri.txt'
    scores_without_ex_second = calculate_average_score_with_std(file_name=file_name,trial_number=trial_number,first=first,second=1-first)
    plot_hist(scores_without_ex_second,file_name,end=10)
    