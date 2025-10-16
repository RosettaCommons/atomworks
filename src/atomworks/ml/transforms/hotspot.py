import numpy as np
import random
from typing import List, Tuple, Union

class HotspotSampler:
    def __init__(self, 
        hotspot_list: List[List[int]], 
        n_tokens: int,
        is_token_in_crop: List[bool],
        ):
        """
        初始化热点采样器
        
        Args:
            hotspot_list: 热点列表，每个子列表代表一条链的热点位置
                        例如: [[0,30,40], [42,45]] 表示有两条链
            n_tokens: 总token数量
            is_token_in_crop: 布尔列表，表示每个token是否在裁剪区域内
        """
        self.hotspot_list = hotspot_list
        self.num_chains = len(hotspot_list)
        self.n_tokens = n_tokens
        self.is_token_in_crop = np.array(is_token_in_crop, dtype=bool)
        
        # 初始化数据结构
        self.is_hotspot = np.zeros(n_tokens, dtype=bool)
        self.chain_assignments = np.full(n_tokens, -1, dtype=int)
        
        self._process_hotspots()
    
    def _process_hotspots(self):
        """处理热点列表，生成is_hotspot向量和链分配信息"""
        
        # 填充热点信息和链分配
        for chain_idx, chain_hotspots in enumerate(self.hotspot_list):
            for pos in chain_hotspots:
                if pos < self.n_tokens:
                    self.is_hotspot[pos] = True
                    self.chain_assignments[pos] = chain_idx
        
        print(f"总热点数量: {np.sum(self.is_hotspot)}")
        print(f"在裁剪区域内的热点数量: {np.sum(self.is_hotspot & self.is_token_in_crop)}")
    
    def get_available_hotspots_by_chain(self, selected_chains: List[int]) -> np.ndarray:
        """
        获取指定链中且在裁剪区域内的热点掩码
        
        Args:
            selected_chains: 选中的链索引列表
            
        Returns:
            布尔掩码，True表示该位置是选中链的热点且在裁剪区域内
        """
        chain_mask = np.zeros_like(self.is_hotspot, dtype=bool)
        
        for chain_idx in selected_chains:
            # 选中指定链的热点，并且这些热点在裁剪区域内
            chain_mask |= (self.chain_assignments == chain_idx) & self.is_hotspot & self.is_token_in_crop
        
        return chain_mask
    
    def sample(self, all_chains_prob: float, sample_percentage_range: Tuple[float, float] = (0.3, 0.7)) -> np.ndarray:
        """
        根据规则进行热点采样（不实际裁剪，但考虑裁剪约束）
        
        Args:
            all_chains_prob: 选择所有链的概率（0-1之间）
            sample_percentage_range: 采样百分比范围，如(0.3, 0.7)表示30%-70%
            
        Returns:
            采样后的布尔掩码（True表示被采样的热点），长度始终为n_tokens
        """
        # 1. 选择链策略
        use_all_chains = random.random() < all_chains_prob
        
        if use_all_chains:
            # 选择所有链
            selected_chains = list(range(self.num_chains))
            print("策略: 选择所有链进行采样")
        else:
            # 随机选择一条链
            selected_chain = random.randint(0, self.num_chains - 1)
            selected_chains = [selected_chain]
            print(f"策略: 选择链 {selected_chain} 进行采样")
        
        # 2. 获取选中链中且在裁剪区域内的热点
        available_hotspots_mask = self.get_available_hotspots_by_chain(selected_chains)
        available_hotspot_indices = np.where(available_hotspots_mask)[0]
        
        # 3. 随机确定采样百分比
        sample_percentage = random.uniform(sample_percentage_range[0], sample_percentage_range[1])
        
        # 4. 在可用的热点中进行采样
        num_to_sample = max(1, int(len(available_hotspot_indices) * sample_percentage))
        
        if len(available_hotspot_indices) > 0:
            sampled_indices = np.random.choice(
                available_hotspot_indices, 
                size=min(num_to_sample, len(available_hotspot_indices)), 
                replace=False
            )
            sampled_mask = np.zeros_like(self.is_hotspot, dtype=bool)
            sampled_mask[sampled_indices] = True
        else:
            sampled_mask = np.zeros_like(self.is_hotspot, dtype=bool)
            print("警告: 没有可用的热点进行采样")
        
        # 5. 统计信息
        total_available = len(available_hotspot_indices)
        total_hotspots_in_selected_chains = np.sum(
            (self.chain_assignments == selected_chains[0]) & self.is_hotspot
        ) if not use_all_chains else np.sum(self.is_hotspot)
        
        print(f"选中链中的总热点数量: {total_hotspots_in_selected_chains}")
        print(f"在裁剪区域内的可用热点数量: {total_available}")
        print(f"采样百分比: {sample_percentage:.1%}")
        print(f"实际采样数量: {np.sum(sampled_mask)}")
        print(f"采样热点位置: {sampled_indices if len(available_hotspot_indices) > 0 else []}")
        
        return sampled_mask
    
    def get_detailed_sampling_info(self, sampled_mask: np.ndarray) -> dict:
        """获取详细的采样信息"""
        info = {
            'total_sampled': np.sum(sampled_mask),
            'total_available_hotspots': np.sum(self.is_hotspot & self.is_token_in_crop),
            'sampled_positions': np.where(sampled_mask)[0].tolist(),
            'chain_breakdown': {},
            'availability_breakdown': {
                'total_hotspots': np.sum(self.is_hotspot),
                'hotspots_in_crop': np.sum(self.is_hotspot & self.is_token_in_crop),
                'hotspots_outside_crop': np.sum(self.is_hotspot & ~self.is_token_in_crop)
            }
        }
        
        for chain_idx in range(self.num_chains):
            chain_hotspots = (self.chain_assignments == chain_idx) & self.is_hotspot
            chain_hotspots_in_crop = chain_hotspots & self.is_token_in_crop
            chain_sampled = sampled_mask & chain_hotspots
            
            info['chain_breakdown'][f'chain_{chain_idx}'] = {
                'total_hotspots': np.sum(chain_hotspots),
                'hotspots_in_crop': np.sum(chain_hotspots_in_crop),
                'sampled_count': np.sum(chain_sampled),
                'sampled_positions': np.where(chain_sampled)[0].tolist()
            }
        
        return info

# 使用示例
if __name__ == "__main__":
    # 1. 初始化热点采样器
    hotspot_list = [[0, 30, 40, 55, 60, 75], [42, 45, 50, 65, 80], [10, 20, 35, 70]]
    n_tokens = 100
    
    # 创建裁剪向量（示例：随机裁剪）
    rng = np.random.default_rng(42)  # 固定随机种子用于可重复性
    is_token_in_crop = rng.random(n_tokens) > 0.3  # 70%的位置在裁剪区域内
    
    sampler = HotspotSampler(hotspot_list, n_tokens, is_token_in_crop)
    
    # 2. 进行采样（70%概率选择所有链，30%概率选择随机一条链）
    sampled_mask = sampler.sample(all_chains_prob=0.7, sample_percentage_range=(0.4, 0.6))
    
    # 3. 获取详细采样信息
    sampling_info = sampler.get_detailed_sampling_info(sampled_mask)
    
    print("\n详细采样信息:")
    print(f"总采样数量: {sampling_info['total_sampled']}")
    print(f"可用热点总数: {sampling_info['total_available_hotspots']}")
    print(f"采样位置: {sampling_info['sampled_positions']}")
    
    print("\n可用性分析:")
    avail_info = sampling_info['availability_breakdown']
    print(f"总热点: {avail_info['total_hotspots']}")
    print(f"裁剪区域内热点: {avail_info['hotspots_in_crop']}")
    print(f"裁剪区域外热点: {avail_info['hotspots_outside_crop']}")
    
    print("\n各链分布:")
    for chain, chain_info in sampling_info['chain_breakdown'].items():
        print(f"{chain}: 总热点{chain_info['total_hotspots']}, "
              f"裁剪内{chain_info['hotspots_in_crop']}, "
              f"采样{chain_info['sampled_count']}")
