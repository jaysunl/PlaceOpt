import sys
from collections import deque
import math
#import torch
#import numpy as np
#import what you need


def read_file(file_path):
    with open(file_path, 'rb') as file:
        text = file.read().decode('utf-8')

    def parse_line(line):
        line = line.strip()
        if not line:
            return None
        if not (line.startswith('(') and line.endswith(')')):
            raise ValueError(f"invalid line format: {line}")
        body = line[1:-1]
        parts = []
        depth = 0
        start = 0
        for idx, ch in enumerate(body):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(body[start:idx].strip())
                start = idx + 1
        parts.append(body[start:].strip())
        if len(parts) != 3:
            raise ValueError(f"invalid tuple size: {line}")

        in_point = int(parts[0])
        out_point = int(parts[2])
        mid_str = parts[1]
        if mid_str == 'None':
            mid = None
        else:
            if not (mid_str.startswith('(') and mid_str.endswith(')')):
                raise ValueError(f"invalid middle tuple: {line}")
            inner = mid_str[1:-1]
            inner_parts = [p.strip() for p in inner.split(',')]
            if len(inner_parts) != 2:
                raise ValueError(f"invalid middle tuple size: {line}")
            kind = inner_parts[0]
            if (kind.startswith("'") and kind.endswith("'")) or (kind.startswith('"') and kind.endswith('"')):
                kind = kind[1:-1]
            count = int(inner_parts[1])
            mid = (kind, count)
        return (in_point, mid, out_point)

    results = []
    for raw_line in text.splitlines():
        item = parse_line(raw_line)
        if item is not None:
            results.append(item)
    return tuple(results)



#TODO
def b2i_transform(buffer_list, buffer_cost: float = 1, inverter_cost: float = 0.5):
    # Assumption: 1. the first buffer start at point 1
    #             2. the first element in tuple is at the source side, the last element is at the sink side

    if not buffer_list:
        raise ValueError("empty buffer list")
        return buffer_list

    depth = 0
    tree_roots = []  # depth 0
    netlist = {}  # net index -> source, load 
    segments = {} # segment index -> parent, children, depth

    # first pass: identify roots and create netlist
    for segment_index, buffer in enumerate(buffer_list):
        in_point, mid, out_point = buffer
        
        # find roots
        if in_point == 1:
            tree_roots.append(segment_index)
        
        # create netlist
        if in_point not in netlist:
            netlist[in_point] = {'source': None, 'load': [segment_index]}
        else:
            netlist[in_point]['load'].append(segment_index)
        if out_point not in netlist:
            netlist[out_point] = {'source': segment_index, 'load': []}
        else:
            raise ValueError("multiple drivers detected!")
    
    # second pass: create segments
    for segment_index, buffer in enumerate(buffer_list):
        in_point, mid, out_point = buffer
        
        # create segments
        parent = netlist[in_point]['source']
        children = netlist[out_point]['load']
        depth = -1

        segments[segment_index] = {'parent': parent, 'children': children, 'depth': depth}
    
    # third pass: assign depth using BFS
    traverse_order = []
    queue = deque()
    for root in tree_roots:
        segments[root]['depth'] = 0
        queue.append(root)

    while queue:
        current = queue.popleft()
        current_depth = segments[current]['depth']
        
        queue.extend(segments[current]['children'])
        for child in segments[current]['children']:
            segments[child]['depth'] = current_depth + 1
            # print("parent:", current, "child:", child, "depth:", segments[child]['depth'])
        
        traverse_order.append(current)

    traverse_order.reverse()  # from leaves to roots

    # update dp table according to traverse order
    dp = [[-1, -1] for _ in range(len(buffer_list))]
    for segment_index in traverse_order:
        in_point, mid, out_point = buffer_list[segment_index]
        children = segments[segment_index]['children']

        # base case: leaf node
        if len(children) == 0:
            if mid is None:
                dp[segment_index][1] = 0
                dp[segment_index][0] = inverter_cost
            else:    # buffer exists
                if mid[1] % 2:  # odd
                    dp[segment_index][1] = (mid[1] - 1) * inverter_cost + buffer_cost
                    dp[segment_index][0] = mid[1] * inverter_cost
                else:           # even
                    dp[segment_index][1] = mid[1] * inverter_cost
                    dp[segment_index][0] = (mid[1] - 1) * inverter_cost + buffer_cost
        # non-leaf node
        else:
            cost_change_polarity = -1
            cost_same_polarity = -1
            if mid is None:
                cost_same_polarity = 0
                cost_change_polarity = inverter_cost
            else:
                if mid[1] % 2:  # odd
                    cost_same_polarity = (mid[1] - 1) * inverter_cost + buffer_cost
                    cost_change_polarity = mid[1] * inverter_cost
                else:           # even
                    cost_same_polarity = mid[1] * inverter_cost
                    cost_change_polarity = (mid[1] - 1) * inverter_cost + buffer_cost
            
            # accumulate children's cost
            children_polarity_1 = 0
            children_polarity_0 = 0
            for child in children:
                children_polarity_1 += dp[child][1]
                children_polarity_0 += dp[child][0]
            
            dp[segment_index][1] = min(cost_same_polarity + children_polarity_1,
                                        cost_change_polarity + children_polarity_0)
            dp[segment_index][0] = min(cost_same_polarity + children_polarity_0,
                                        cost_change_polarity + children_polarity_1)

    # reconstruct the buffer list
    new_buffer_list = []
    queue = deque()
    for root in tree_roots:
        queue.append((root, 1))  # (segment index, desired polarity)
    
    while queue:
        current, desired_polarity = queue.popleft()
        in_point, mid, out_point = buffer_list[current]
        children = segments[current]['children']

        # leaf node
        if len(children) == 0:
            if mid is None:
                if desired_polarity == 1:
                    new_buffer_list.append((in_point, None, out_point))
                else:
                    new_buffer_list.append((in_point, ('i', 1), out_point))
            else:
                if desired_polarity == 1:
                    if mid[1] == 1:  
                        new_buffer_list.append((in_point, ('b', 1), out_point))
                    elif mid[1] % 2:  # odd
                        new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
                    else:           # even
                        new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                else:  # desired polarity 0
                    if mid[1] == 1:  
                        new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                    elif mid[1] % 2:  # odd
                        new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                    else:           # even
                        new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
        # non-leaf node
        else:
            cost_change_polarity = -1
            cost_same_polarity = -1
            if mid is None:
                cost_same_polarity = 0
                cost_change_polarity = inverter_cost
            else:
                if mid[1] % 2:  # odd
                    cost_same_polarity = (mid[1] - 1) * inverter_cost + buffer_cost
                    cost_change_polarity = mid[1] * inverter_cost
                else:           # even
                    cost_same_polarity = mid[1] * inverter_cost
                    cost_change_polarity = (mid[1] - 1) * inverter_cost + buffer_cost
            
            if desired_polarity == 1:
                if dp[current][1] == cost_same_polarity + sum(dp[child][1] for child in children):
                    if mid is None:
                        new_buffer_list.append((in_point, None, out_point))
                    else:
                        if mid[1] == 1:
                            new_buffer_list.append((in_point, ('b', 1), out_point))
                        elif mid[1] % 2:  # odd
                            new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
                        else:           # even
                            new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                    for child in children:
                        queue.append((child, 1))
                else:
                    if mid is None:
                        new_buffer_list.append((in_point, ('i', 1), out_point))
                    else:
                        if mid[1] == 1:
                            new_buffer_list.append((in_point, ('i', 1), out_point))
                        elif mid[1] % 2:  # odd
                            new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                        else:           # even
                            new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
                            if mid[1] <= 0:
                                raise ValueError("buffer count must be positive")
                    for child in children:
                        queue.append((child, 0))
            else:  # desired polarity 0
                if dp[current][0] == cost_same_polarity + sum(dp[child][0] for child in children):
                    if mid is None:
                        new_buffer_list.append((in_point, None, out_point))
                    else:
                        if mid[1] == 1:
                            new_buffer_list.append((in_point, ('b', 1), out_point))
                        elif mid[1] % 2:  # odd
                            new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
                        else:           # even
                            new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                    for child in children:
                        queue.append((child, 0))
                else:
                    if mid is None:
                        new_buffer_list.append((in_point, ('i', 1), out_point))
                    else:
                        if mid[1] == 1:
                            new_buffer_list.append((in_point, ('i', 1), out_point))
                        elif mid[1] % 2:  # odd
                            new_buffer_list.append((in_point, ('i', mid[1]), out_point))
                        else:           # even
                            new_buffer_list.append((in_point, (('i', mid[1] - 1), ('b', 1)), out_point))
                            if mid[1] <= 0:
                                raise ValueError("buffer count must be positive")
                    for child in children:
                        queue.append((child, 1))
    # print("dp table:", dp)
    return tuple(new_buffer_list)

def polarity_check(buffer_list):
    """Return True if even number of inverters from source to sink for all paths."""
    if not buffer_list:
        return True

    # Build netlist and segments structure
    netlist = {}
    segments = {}
    tree_roots = []

    for segment_index, buffer in enumerate(buffer_list):
        in_point, mid, out_point = buffer

        if in_point == 1:
            tree_roots.append(segment_index)

        if in_point not in netlist:
            netlist[in_point] = {'source': None, 'load': [segment_index]}
        else:
            netlist[in_point]['load'].append(segment_index)
        if out_point not in netlist:
            netlist[out_point] = {'source': segment_index, 'load': []}
        else:
            if netlist[out_point]['source'] is not None:
                raise ValueError("multiple drivers detected!")
            netlist[out_point]['source'] = segment_index

    for segment_index, buffer in enumerate(buffer_list):
        in_point, mid, out_point = buffer
        parent = netlist[in_point]['source']
        children = netlist[out_point]['load']
        segments[segment_index] = {'parent': parent, 'children': children}

    # Traverse from source to all sinks and check polarity
    queue = deque()
    for root in tree_roots:
        queue.append((root, 0))  # (segment index, inverter count)

    while queue:
        current, inv_count = queue.popleft()
        in_point, mid, out_point = buffer_list[current]

        # Count inverters in current segment
        current_inv_count = 0
        if mid is not None:
            if isinstance(mid, tuple) and len(mid) == 2:
                if isinstance(mid[0], str):
                    # Single type: ('i', count) or ('b', count)
                    if mid[0] == 'i':
                        current_inv_count = mid[1]
                elif isinstance(mid[0], tuple):
                    # Nested tuple: (('i', count1), ('b', count2))
                    for item in mid:
                        if item[0] == 'i':
                            current_inv_count += item[1]

        total_inv_count = inv_count + current_inv_count
        children = segments[current]['children']

        # If leaf node (sink), check polarity
        if len(children) == 0:
            if total_inv_count % 2 != 0:
                return False
        else:
            for child in children:
                queue.append((child, total_inv_count))
    return True


#assume the cost of a buffer is 1, the cost of an inverter is 0 < a < 1


def main():
    if len(sys.argv) != 2:
        print("usage: ./b2i_transform.py ./file")
        return 1
    path = sys.argv[1]
    buffer_list = read_file(path)

    result = b2i_transform(buffer_list)
    print("old buffer list:", buffer_list)
    print("new buffer list:", result)

    if not polarity_check(result):
        print("polarity check failed!")
    else:
        print("polarity check passed!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
