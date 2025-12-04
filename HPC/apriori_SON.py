from pyspark.sql import SparkSession
from collections import defaultdict
from itertools import combinations
import math
import time
import csv
import random
import os


class ModifyDataset:
    """
    A utility class for modifying Spark RDDs of transactions based on various criteria.
    """

    @staticmethod
    def change_distinct_items(transactions_rdd, max_distinct_items):
        """
        Modifies the dataset to achieve a specific number of distinct items by filtering transactions.
        """
        # 1. Get top-N frequent items
        item_counts = transactions_rdd.flatMap(lambda x: x).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
        top_items = item_counts.takeOrdered(max_distinct_items, key=lambda x: -x[1])  # top-N by frequency
        top_items_dict = dict(top_items)
        top_items_list = list(top_items_dict.keys())  # for sampling

        # Broadcast the top items for efficiency
        sc = transactions_rdd.context
        broadcast_top_items = sc.broadcast(top_items_dict)
        broadcast_top_items_list = sc.broadcast(top_items_list)

        # Function to transform each transaction
        def transform_transaction(transaction):
            top_items_set = broadcast_top_items.value
            top_items_pool = broadcast_top_items_list.value

            # Keep only frequent items
            filtered = [item for item in transaction if item in top_items_set]
            missing = len(transaction) - len(filtered)

            # Add replacements to preserve length
            if missing > 0:
                replacements = random.choices(top_items_pool, k=missing)
                filtered.extend(replacements)

            return filtered

        # use mao to appy the function to all transactions
        transformed_rdd = transactions_rdd.map(transform_transaction)

        return transformed_rdd

    @staticmethod
    def increase_avg_length(transactions_rdd, target_avg_len):
        """
        Pads transactions to reach a target average length without increasing the number of distinct items.
        """

        # 1. Extract different item
        item_set = transactions_rdd.flatMap(lambda x: x).distinct().collect()
        sc = transactions_rdd.context
        broadcast_item_set = sc.broadcast(item_set)

        def increase_transaction(transaction):
            current_len = len(transaction)
            target_len = target_avg_len

            if current_len < target_len:
                padding_items = random.choices(broadcast_item_set.value, k=target_len - current_len)
                return frozenset(list(transaction) + padding_items)
            else:
                return transaction

        padded_rdd = transactions_rdd.map(increase_transaction)
        return padded_rdd


    @staticmethod
    def filter_dataset_size(transactions_rdd, max_size_percentage):
        """
        Filters transactions to only include a `max_size_percentage` of the dataset.
        """
        total_transactions = transactions_rdd.count()
        num_transactions_to_keep = int(total_transactions * max_size_percentage)
        
        # Take the first `num_transactions_to_keep` transactions by zipping with index
        filtered_transactions = transactions_rdd.zipWithIndex().filter(
            lambda x: x[1] < num_transactions_to_keep
        ).keys() # .keys() gets rid of the index

        return filtered_transactions


class SONUtils:
    """
    Utility class containing helper methods for the SON algorithm.
    """

    @staticmethod
    def get_set_of_ints(line):
        """
        Converts a line of space-separated integers into a frozenset of integers.
        Frozenset is used to make sets hashable for use as dictionary keys.
        """
        return frozenset(map(int, line.split()))


    @staticmethod
    def count_candidates(partition, candidates):
        """
        Counts the occurrences of candidate itemsets in a given partition.
        """
        partition = list(partition) # Convert iterator to list for multiple passes
        candidate_counts = defaultdict(int)

        for transaction in partition:
            # For each transaction, check against all candidates
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        return list(candidate_counts.items())
    

    @staticmethod
    def generate_candidates(L_k_minus_1, k):
        """
        Generates candidate itemsets of size k from frequent itemsets of size k-1
        using the Apriori join and prune steps.
        """

        candidates = set()
        # Convert to list and sort for consistent joining logic (Apriori's join property)
        L_k_minus_1_list = sorted(list(L_k_minus_1)) 
        L_k_minus_1_set = set(L_k_minus_1) # Keep as set for efficient pruning lookup

        for i in range(len(L_k_minus_1_list)):
            for j in range(i + 1, len(L_k_minus_1_list)):
                l1 = list(L_k_minus_1_list[i])
                l2 = list(L_k_minus_1_list[j])
                
                # Apriori join step: (k-1)-itemsets {i1, ..., ik-1} and {i1, ..., ik-2, ik'}
                # can be joined if their first k-2 items are identical.
                # For k=2, this condition is trivially true (no first k-2 items).
                if k == 2 or l1[:-1] == l2[:-1]: # Compare all but the last item
                    candidate = frozenset(L_k_minus_1_list[i] | L_k_minus_1_list[j])

                    # Ensure the candidate has the correct size k (union might be smaller if not distinct)
                    if len(candidate) == k:
                        # Apriori prune step: all (k-1)-subsets of the candidate must be frequent.
                        # If any subset is not frequent, the candidate cannot be frequent.
                        if all(frozenset(subset) in L_k_minus_1_set for subset in combinations(candidate, k-1)):
                            candidates.add(candidate)
        return candidates


    @staticmethod
    def apriori_local(partition, min_supp_local):
        """
        Runs the Apriori algorithm locally on a given partition of transactions.
        """
        transactions = list(partition) # Convert iterator to list for multiple passes
        item_counts = defaultdict(int)

        # First pass: Count occurrences of each single item
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # L1: Frequent singletons (itemsets of size 1)
        L1 = {frozenset([item]) for item, count in item_counts.items() if count >= min_supp_local}

        frequent_itemsets = list(L1) # Stores all frequent itemsets found
        L_k_minus_1 = set(L1) # Used to generate candidates for the next iteration

        # Initialize candidate statistics
        candidate_stats = {}
        # Count of distinct items seen in the partition (potential k=1 candidates)
        candidate_stats[1] = len(item_counts) 

        # Filter transactions: remove items that are not frequent singletons (optimization)
        frequent_singletons_set = {item for fs in L1 for item in fs}
        transactions_filtered = [
            frozenset(item for item in transaction if item in frequent_singletons_set)
            for transaction in transactions]
        
        # Remove empty transactions that might result empty from filtering
        transactions_filtered = [t for t in transactions_filtered if len(t) > 0]

        k = 2
        while L_k_minus_1:
            # Generate candidates of size k from L_{k-1}
            candidates = SONUtils.generate_candidates(L_k_minus_1, k)
            candidate_stats[k] = len(candidates)

            if not candidates:
                break # No more candidates, so stop

            # Count occurrences of candidates in the filtered transactions
            candidate_counts_k = defaultdict(int)
            for transaction in transactions_filtered:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts_k[candidate] += 1

            # L_k: Frequent itemsets of size k
            L_k = {candidate for candidate, count in candidate_counts_k.items() if count >= min_supp_local}

            frequent_itemsets.extend(L_k) # Add to the overall list of frequent itemsets

            # For the next iteration, L_k becomes L_{k-1}
            L_k_minus_1 = set(L_k) 
            
            k += 1
        return frequent_itemsets, candidate_stats
        

def SON(transactions_rdd, support, num_partitions, input_file):
    """
    Implements the SON (Savasere, Omiecinski, and Navathe) algorithm for frequent itemset mining.
    """
    # Calculate initial dataset statistics
    num_transactions = transactions_rdd.count()
    
    transaction_lengths = transactions_rdd.map(lambda x: len(x))
    total_length = transaction_lengths.reduce(lambda a, b: a + b)
    average_transaction_length = total_length / num_transactions
    
    distinct_items = transactions_rdd.flatMap(lambda x: x).distinct().count()
    
    dataset_density = average_transaction_length / distinct_items if distinct_items > 0 else 0

    print(f"\n--- Running SON for {input_file} ---")
    print(f"Number of transactions: {num_transactions}")
    print(f"Average transaction length: {average_transaction_length:.2f}")
    print(f"Number of distinct items: {distinct_items}")
    print(f"Dataset density: {dataset_density:.6f}")
    print(f"Running with minimum support: {support}")
    print(f"Number of partitions: {num_partitions}")

    # Calculate global and local minimum support counts
    min_supp = math.ceil(support * num_transactions)
    min_supp_local = math.ceil(min_supp / num_partitions)
    print(f"Global minimum support count: {min_supp}")
    print(f"Local minimum support count: {min_supp_local}")


    # --- Phase 1: Local frequent itemsets ---
    start = time.time()

    # Repartition the RDD for parallel processing
    transactions_rdd_partitioned = transactions_rdd.repartition(num_partitions)
    
    # Apply local Apriori to each partition
    # mapPartitions returns an iterator for each partition, which contains the results 
    # from apriori_local (frequent_itemsets, candidate_stats)
    local_results = transactions_rdd_partitioned.mapPartitions(
        lambda partition: [SONUtils.apriori_local(partition, min_supp_local)]
    ).collect() # Collect results from all partitions

    all_frequent_itemsets_from_local = []
    candidate_stats_all = defaultdict(int)  # Aggregated candidate counts from all partitions

    for frequent_itemsets_local, candidate_stats_local in local_results:
        all_frequent_itemsets_from_local.extend(frequent_itemsets_local)
        for k, count in candidate_stats_local.items():
            candidate_stats_all[k] += count
            
    # Remove duplicates to get the set of all unique candidate itemsets from Phase 1
    # A Python set is used because it can store only unique elements
    candidates_for_phase2 = set(all_frequent_itemsets_from_local)    

    # --- Phase 2: Global verification ---
    # Broadcast candidates to all executors for efficient lookup
    candidate_broadcast = sc.broadcast(candidates_for_phase2)
    
    # Count occurrences of broadcasted candidates in the entire dataset
    global_counts = transactions_rdd_partitioned.mapPartitions(
        lambda partition: SONUtils.count_candidates(partition, candidate_broadcast.value)
    )
    
    # Sum counts for each candidate and filter by global minimum support
    apriori_frequent_itemsets = global_counts.reduceByKey(lambda x, y: x + y)\
                                            .filter(lambda x: x[1] >= min_supp)\
                                            .keys().collect() # Get only the itemsets, not their counts

    end = time.time()
    total_time = end - start
    print(f"SON execution time: {total_time:.2f} seconds")

    print(f"Number of global frequent itemsets found: {len(apriori_frequent_itemsets)}")

    print("\n=== Candidate Counts by k (Aggregated from local Apriori runs) ===")
    for k in sorted(candidate_stats_all.keys()):
        print(f"k={k}: {candidate_stats_all[k]} candidates")

    # Write to CSV continuously
    with open('SON_experiment_results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            input_file, num_transactions, average_transaction_length, distinct_items, dataset_density,
            support, num_partitions, min_supp_local, len(apriori_frequent_itemsets), total_time
        ])
        for k in sorted(candidate_stats_all.keys()):
            writer.writerow([f"k={k}: {candidate_stats_all[k]} candidates"])



if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("SON_Experiments").getOrCreate()
    sc = spark.sparkContext

    input_file = "mushroom.txt"  # Replace with your input file path

   # Write CSV header only once
    with open('SON_experiment_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'dataset', 'num_transactions', 'average_transaction_length', 'distinct_items', 'dataset_density',
            'support', 'num_partitions', 'min_supp_local', 'len(apriori_frequent_itemsets)', 'Time'
        ])
        writer.writerow(["Candidate Counts by k (following each run)"])  # separator

    # --- Experiment 1.: Varying Support ---
    print("\n--- Running Experiment 1: Varying Support ---")
    support_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
    for support in support_values:
        transactions_rdd_base = sc.textFile(input_file).map(SONUtils.get_set_of_ints).persist()

        SON(transactions_rdd_base, support=support, num_partitions=15, input_file=input_file)
        transactions_rdd_base.unpersist()
    

    # Experiment 2: Varying Transaction Length
    print("\n--- Running Experiment 2: Varying Transaction Length ---")
    avg_lengths_to_truncate = [25, 30, 35, 40, 45, 50]
    for max_len in avg_lengths_to_truncate:
        transactions_rdd_base = sc.textFile(input_file).map(SONUtils.get_set_of_ints).persist()
        modified_rdd = ModifyDataset.increase_avg_length(transactions_rdd_base, target_avg_len=max_len)
        SON(modified_rdd, support=0.25, num_partitions=15, input_file=f"{input_file}_len_{max_len}")
        transactions_rdd_base.unpersist()

    # --- Experiment 3: Varying Dataset Size ---
    print("\n--- Running Experiment 3: Varying Dataset Size ---")
    dataset_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for size_percentage in dataset_sizes:
        transactions_rdd_base = sc.textFile(input_file).map(SONUtils.get_set_of_ints).persist()
        transactions_rdd_filtered = ModifyDataset.filter_dataset_size(transactions_rdd_base, size_percentage)
        SON(transactions_rdd_filtered, support=0.25, num_partitions=15, input_file=input_file)
        transactions_rdd_base.unpersist()

    # --- Experiment 4: Varying Distinct Items (Implicitly affects density) ---
    print("\n--- Running Experiment 4: Varying Distinct Items ---")
    distinct_items_to_filter = [100, 75, 50, 25, 20, 15]
    for max_items in distinct_items_to_filter:
        transactions_rdd_base = sc.textFile(input_file).map(SONUtils.get_set_of_ints).persist()
        modified_rdd = ModifyDataset.change_distinct_items(transactions_rdd_base, max_distinct_items=max_items)

        SON(modified_rdd, support=0.25, num_partitions=15, input_file=input_file)
        transactions_rdd_base.unpersist()

    # --- Experiment 5: Varying Density by adjusting Length and Distinct Items ---
    print("\n--- Running Experiment 5: Varying Density by adjusting Length and Distinct Items ---")
    
    # Define the parameters for the experiments
    max_distinct_items = [80]
    min_lengths = [20, 25, 30, 35, 40, 45, 50]  # Target average lengths for transactions
    for max_items in max_distinct_items:
        for min_length in min_lengths:
            transactions_rdd_base = sc.textFile(input_file).map(SONUtils.get_set_of_ints).persist()

            transactions_rdd_base = ModifyDataset.increase_avg_length(transactions_rdd_base, target_avg_len=min_length)
            modified_rdd = ModifyDataset.change_distinct_items(transactions_rdd_base, max_distinct_items=max_items)

            SON(modified_rdd, support=0.25, num_partitions=15, input_file=input_file)
            transactions_rdd_base.unpersist()


    spark.stop()