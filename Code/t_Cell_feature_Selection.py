import numpy as np
# import pandas as pd
import Bio.SeqUtils.ProtParam as prot
from collections import Counter
# from pyclone.cluster import _count_clones
from scipy.stats import entropy

from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import seq3
import csv


# Load Data
data_path = "/alina-data1/Zara/TCell_Receptor/Embedding_Generation/"
data_name = "splitted_sequences_50k_1"

seq_data = np.loadtxt(data_path+ data_name + ".npy",dtype=str)

sequence_val_all = seq_data.tolist()

seq_replaced = []
for sequ in sequence_val_all:
    seq_replaced.append(sequ.replace('X', 'A'))

all_embeddings = []
for u in range(len(seq_replaced)):
    sequence_val = [seq_replaced[u]]

    tcr_seqs = [sequence_val[0]]
    print("tcr_seqs: ",tcr_seqs)

    #########################################################################################
    # CDR3 length
    #cdr3_lengths = [len(seq[seq.index('C') + 1: seq.index('F')]) for seq in tcr_seqs]
    cdr3_lengths = [len(seq[seq.index('C') + 1: -1]) if seq[-1]=='F' else len(seq[seq.index('C') + 1: seq.index('V')]) for seq in tcr_seqs]

#     print("CDR3 lengths:", cdr3_lengths)
    #########################################################################################


    #########################################################################################
    # Amino acid composition
    aa_compositions = [Counter(seq) for seq in tcr_seqs]
    aa_freqs = [dict((k, v / sum(aa_comp.values())) for k, v in aa_comp.items()) for aa_comp in aa_compositions]
#     print("Amino acid compositions:", aa_freqs)
    
    ###############################
    # Define a dictionary with all possible amino acids
    all_aa = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

    # Initialize the counts to zero for all possible amino acids
    aa_freq_temp = {aa: 0 for aa in all_aa}
    dict_org = (aa_freqs[0])
    dict_org_vals = list(dict_org.values())
    dict_org_keys = list(dict_org.keys())

    for dic_v in range(len(dict_org)):
        aa_freq_temp[dict_org_keys[dic_v]] = dict_org_vals[dic_v]
    ###############################
    #########################################################################################
    

    #########################################################################################
    # Hydrophobicity and charge
    hydrophobicity_scores = [ProtParam.ProteinAnalysis(seq).gravy() for seq in tcr_seqs]
    charge_scores = [ProtParam.ProteinAnalysis(seq).charge_at_pH(7.4) for seq in tcr_seqs]
    # print("Hydrophobicity scores:", hydrophobicity_scores)
    # print("Charge scores:", charge_scores)

    #########################################################################################
    
    
    #########################################################################################
    # Charge distribution
    pos_charge_counts = [sum([1 for aa in seq if aa in ['R', 'K', 'H']]) for seq in tcr_seqs]
    neg_charge_counts = [sum([1 for aa in seq if aa in ['D', 'E']]) for seq in tcr_seqs]
    charge_distribution = [(pos_count, neg_count) for pos_count, neg_count in zip(pos_charge_counts, neg_charge_counts)]
    # print("Charge distribution:", charge_distribution)

    #########################################################################################
    
    
    #########################################################################################
    # CDR3 sequence motif
    cdr3_motifs = [seq[seq.index('G') + 1: -1] if 'G' in seq  else '-' for seq in tcr_seqs]
    # print("CDR3 motifs:", cdr3_motifs)

    # Load the Blosum62 matrix from a file
    # blosum62_df = pd.read_csv("E:/RA/T_Cell_Features/Data/blosum62.csv", index_col=0)
    blosum62_df = []
    with open('/alina-data2/Sarwan/T_Cell_Features_Selection/Dataset/blosum62.csv', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            blosum62_df.append(row)

    blosum62_df = np.array(blosum62_df)   

    arr = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']

    # Define the motif sequence
    # motif = "VGXGXG"
    motif = cdr3_motifs[0]

    # Compute the similarity score for each pair of amino acids in the motif
    score_sum = 0
    for i in range(len(motif)-1):
        aa1 = motif[i]
        index1 = arr.index(aa1) 

        aa2 = motif[i+1]
        index2 = arr.index(aa2) 

        score = int(blosum62_df[index1, index2])
        score_sum += score

    # Compute the average similarity score for the motif
    if score_sum != 0:
        score_avg = score_sum / (len(motif)-1)
    else:
        score_avg = score_sum / 1

    #print("Motif:", motif)
    #print("Average Motif similarity score:", score_avg)
    #########################################################################################

    #################################################################################
    # Diversity metrics
    seq = tcr_seqs[0]
    # Counting the number of unique TCR sequences for the current sequence
    unique_count = len(set(seq))
    # Computing Shannon entropy of TCR repertoire for the current sequence
    shannon_entropy = entropy(list(Counter(seq).values()), base=2)
    # Computing Simpson index of TCR repertoire for the current sequence
    simpson_index = 1 - sum([(count / len(seq)) ** 2 for tcr, count in Counter(seq).items()])
    # print("Sequence:", seq)
    # print("Shannon entropy:", shannon_entropy)
    # print("Simpson index:", simpson_index)
    #################################################################################

    final_features = np.concatenate([cdr3_lengths,
                    list(aa_freq_temp.values()), 
                    hydrophobicity_scores,
                    charge_scores,
                    list(charge_distribution[0]), 
                    [score_avg], 
                    [shannon_entropy], 
                    [simpson_index]
                   ])

    all_embeddings.append(np.array(final_features))
#     print(final_features)
np.save('all_embeddings.npy', all_embeddings)
print("Done!")