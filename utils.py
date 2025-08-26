def detect_loop_pattern(token_list, min_loop_length=3, max_loop_length=50):
    n = len(token_list)
    
    for loop_len in range(min_loop_length, min(max_loop_length, n // 2)):
        for start in range(n - loop_len * 2):
            pattern = token_list[start:start + loop_len]
            
            consecutive_loops = 1
            pos = start + loop_len
            
            while pos + loop_len <= n:
                if token_list[pos:pos + loop_len] == pattern:
                    consecutive_loops += 1
                    pos += loop_len
                else:
                    break
            
            if consecutive_loops >= 3:
                loop_pattern = {
                    'loop_start': start,
                    'loop_length': loop_len,
                    'num_loops': consecutive_loops,
                    'pattern': pattern
                }
                return loop_pattern
    return None