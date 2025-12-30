import numpy as np
from collections import defaultdict

def consecutive_ones_test(matrix):
    """
    Test if a binary matrix has the consecutive ones property (C1P) for its columns.
    If it does, return an ordering of the columns that satisfies C1P.
    """
    from sage.graphs.pq_trees import reorder_sets

    # if there are no rows or no columns, trivially C1P, return identity ordering
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return True, list(range(len(matrix[0]) if len(matrix) > 0 else 0))

    rows, cols = len(matrix), len(matrix[0])

    # Group columns by their row patterns
    pattern_to_columns = defaultdict(list)

    for col_idx in range(cols):
        row_indices = tuple(sorted([row_idx for row_idx in range(rows) if matrix[row_idx][col_idx] == 1]))
        pattern_to_columns[row_indices].append(col_idx)

    # Separate empty columns
    empty_columns = pattern_to_columns.get((), [])
    if () in pattern_to_columns:  # Only delete if it exists
        del pattern_to_columns[()]

    if not pattern_to_columns:
        return True, list(range(cols))  # All columns are empty

    # Create input for reorder_sets (unique patterns only)
    unique_patterns = list(pattern_to_columns.keys())
    unique_pattern_lists = [list(pattern) for pattern in unique_patterns if
                            pattern]  # Convert tuples to lists, skip empty

    # print(f"Unique patterns: {unique_patterns}")
    # print(f"Pattern to columns mapping: {dict(pattern_to_columns)}")

    if not unique_pattern_lists:
        return True, list(range(cols))

    try:
        # Get ordering of unique patterns
        ordered_patterns = reorder_sets(unique_pattern_lists)
        # print(f"Ordered patterns: {ordered_patterns}")

        # Reconstruct column ordering
        column_ordering = []

        for ordered_pattern in ordered_patterns:
            # Convert back to tuple for lookup
            pattern_tuple = tuple(sorted(list(ordered_pattern)))

            # Find corresponding columns
            if pattern_tuple in pattern_to_columns:
                columns_for_pattern = pattern_to_columns[pattern_tuple]
                # Add all columns with this pattern (preserving original order within pattern)
                column_ordering.extend(sorted(columns_for_pattern))

        # Add empty columns at the end
        column_ordering.extend(empty_columns)

        # print(f"Final column ordering: {column_ordering}")
        return True, column_ordering

    except ValueError as e:
        # print(f"reorder_sets failed: {e}")
        return False, None