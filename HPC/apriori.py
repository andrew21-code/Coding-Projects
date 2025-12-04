from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from itertools import combinations
import time
import matplotlib.pyplot as plt
import math

def get_set_of_ints(line):
    return set(map(int, line.split()))

def get_candidates_in_tr(tr_set, C_k):
    tr_set = set(tr_set)
    return [c for c in C_k.value if set(c).issubset(tr_set)]

def generate_candidates(L_k_minus_1, k): 
    L_k_minus_1_list = list(L_k_minus_1)
    L_k_minus_1_set = {frozenset(i) for i in L_k_minus_1_list}
    C_k = set()

    for i in range(len(L_k_minus_1_list)):
        for j in range(i + 1, len(L_k_minus_1_list)):
            part1, part2 = L_k_minus_1_list[i], L_k_minus_1_list[j]
            if part1[:k-2] == part2[:k-2]:
                candidate = sorted(set(part1).union(part2))
                if len(candidate) == k:
                    is_valid = all(frozenset(subset) in L_k_minus_1_set for subset in combinations(candidate, k-1))
                    if is_valid:
                        C_k.add(tuple(candidate))
    return C_k

# Driver-side Apriori (full scan)
def apriori(transactions_rdd, min_supp, sc):
    print("\n--- Running Apriori ---")
    items = transactions_rdd.flatMap(lambda line: line)
    itemCounts = items.map(lambda item: (item, 1)).reduceByKey(lambda x, y: x + y)
    freqItemCounts = itemCounts.filter(lambda ic: ic[1] >= min_supp)
    L1_items = set(freqItemCounts.keys().collect())
    L1 = sorted([(item,) for item in L1_items])

    L1_bcast = sc.broadcast(L1_items)
    transactions_rdd = transactions_rdd.map(lambda tr: tr.intersection(L1_bcast.value)).persist()

    L_k = L1
    k = 2
    all_frequent_itemsets = []
    C_k_dict = {1: len(L1)}
    all_frequent_itemsets.extend(L1)

    while len(L_k) > 0:
        print(f"\n--- Iteration k={k} ---")
        C_k = generate_candidates(L_k, k)
        print(f"Generated C_k size: {len(C_k)}")
        C_k_dict[k] = len(C_k)

        C_k_broadcast = sc.broadcast(C_k)
        itemsets = transactions_rdd.flatMap(lambda tr: get_candidates_in_tr(tr, C_k_broadcast))
        itemsetCounts = itemsets.map(lambda itemset: (itemset, 1)).reduceByKey(lambda x, y: x + y)
        freqsCounts = itemsetCounts.filter(lambda ic: ic[1] >= min_supp)

        L_k = freqsCounts.keys().collect()
        print(f"Found L_k size: {len(L_k)}")
        all_frequent_itemsets.extend(L_k)
        k += 1

    print("\nApriori execution finished.")
    print("C_k counts per iteration:", C_k_dict)

    # Plot
    plt.figure()
    plt.plot(list(C_k_dict.keys()), list(C_k_dict.values()), marker='o')
    plt.xlabel("k")
    plt.ylabel("Number of Candidates")
    plt.title("Candidate Growth Over Iterations")
    plt.show()
    plt.savefig("apriori_candidate_growth.png")

    return all_frequent_itemsets


def compare_results(apriori_frequent_itemsets, fpgrowth_frequent_itemsets):
    apriori_set = {frozenset(itemset) for itemset in apriori_frequent_itemsets}
    fpgrowth_set = {frozenset(row.items) for row in fpgrowth_frequent_itemsets}

    if apriori_set == fpgrowth_set:
        print("Apriori and FP-Growth results are equal.")
    else:
        print("Apriori and FP-Growth results are NOT equal.")
        print("Apriori-only:", apriori_set - fpgrowth_set)
        print("FP-Growth-only:", fpgrowth_set - apriori_set)
        

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Apriori and FP-Growth Optimized").getOrCreate()
    sc = spark.sparkContext

    support = 2900
    input_file = "chess.txt"

    print(f"\nRunning with minimum support: {support}")
    min_supp = support

    transactions_rdd = sc.textFile(input_file).map(get_set_of_ints).persist()
    num_transactions = transactions_rdd.count()
    avg_len = transactions_rdd.map(len).mean()

    print(f"Number of transactions: {num_transactions}")
    print(f"Average transaction length: {avg_len:.2f}")

    # --- Run Apriori ---
    start = time.time()
    apriori_frequent_itemsets = apriori(transactions_rdd, min_supp, sc)
    end = time.time()
    print(f"Apriori execution time: {end - start:.2f} seconds")
    print(f"Number of frequent itemsets found: {len(apriori_frequent_itemsets)}")

    # --- Run FP-Growth ---
    print("\nRunning FP-Growth...")
    min_support_fp_growth = min_supp / float(num_transactions)
    transactions_df = transactions_rdd.map(lambda tr: Row(items=list(tr))).toDF()
    fp_growth = FPGrowth(itemsCol="items", minSupport=min_support_fp_growth, minConfidence=0.8)
    model = fp_growth.fit(transactions_df)
    fpgrowth_frequent_itemsets = model.freqItemsets.collect()

    # --- Compare results ---
    compare_results(apriori_frequent_itemsets, fpgrowth_frequent_itemsets)

    spark.stop()
