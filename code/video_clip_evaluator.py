#!/usr/bin/env python3
"""
改进的视频CLIP相似度评估器
修复了原始代码中的问题，采用更标准的CLIP处理方式
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import logging
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
from pathlib import Path


def normalize_scenario_id(scenario_id):
    """Return candidate IDs that cover both short and zero-padded directory names."""
    raw = str(scenario_id).strip()
    stripped = raw.lstrip('0') or '0'
    candidates = []
    for value in [raw, stripped, stripped.zfill(6), raw.zfill(6)]:
        if value not in candidates:
            candidates.append(value)
    return candidates


def auto_detect_local_model_path(model_name):
    """Use an already-downloaded Hugging Face snapshot when available."""
    if os.path.exists(model_name):
        return model_name

    if model_name != "openai/clip-vit-base-patch32":
        return model_name

    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots"
    if not snapshot_root.exists():
        return model_name

    for snapshot in sorted(snapshot_root.iterdir()):
        if (snapshot / "config.json").exists() and (snapshot / "pytorch_model.bin").exists():
            return str(snapshot)

    return model_name

class ImprovedVideoCLIPEvaluator:
    def __init__(self, device=None, model_name="openai/clip-vit-base-patch32", local_files_only=False):
        """
        初始化CLIP评估器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            model_name: CLIP模型名称
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = auto_detect_local_model_path(model_name)
        self.local_files_only = local_files_only or os.path.exists(self.model_name)
        
        print(f"使用设备: {self.device}")
        print(f"加载CLIP模型: {self.model_name}")
        
        try:
            # 加载CLIP模型和处理器
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            
            # 设置为评估模式
            self.model.eval()
            
            print("CLIP模型加载完成!")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def extract_frames_from_video(self, video_path, max_frames=100, frame_interval=1):
        """
        从视频中提取帧，返回PIL Image对象列表
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            frame_interval: 帧间隔（每隔几帧取一帧）
        
        Returns:
            frames: PIL Image对象列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # 检查视频是否打开成功
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔frame_interval帧取一帧
                if frame_count % frame_interval == 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转换为PIL Image
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
                    extracted_count += 1
                
                frame_count += 1
                
        finally:
            cap.release()
        
        if not frames:
            raise RuntimeError(f"无法从视频中提取任何帧: {video_path}")
        
        print(f"从 {video_path} 中提取了 {len(frames)} 帧 (总帧数: {total_frames})")
        return frames
    
    def calculate_clip_score(self, video_path, text_prompt, max_frames=100, frame_interval=1):
        """
        计算视频与文本的CLIP相似度 - 标准方式
        
        Args:
            video_path: 视频文件路径
            text_prompt: 文本提示
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
        
        Returns:
            dict: 包含分数信息的字典
        """
        try:
            # 提取视频帧
            frames = self.extract_frames_from_video(video_path, max_frames, frame_interval)
            
            # 使用CLIP processor处理图像和文本
            inputs = self.processor(
                images=frames,
                text=[text_prompt],  # 注意：text需要是列表
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 获取CLIP输出
                outputs = self.model(**inputs)
                
                # 获取图像和文本特征
                image_features = outputs.image_embeds  # [num_frames, embed_dim]
                text_features = outputs.text_embeds    # [1, embed_dim]
                
                # 标准化特征向量
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # 计算相似度分数 (cosine similarity)
                # image_features: [num_frames, embed_dim], text_features: [1, embed_dim]
                similarity_scores = torch.matmul(image_features, text_features.t()).squeeze(-1)
                
                # 转换为numpy数组
                scores = similarity_scores.cpu().numpy()
            
            return {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'max_score': float(np.max(scores)),
                'min_score': float(np.min(scores)),
                'frame_scores': scores.tolist(),
                'num_frames': len(frames),
                'video_path': video_path,
                'text_prompt': text_prompt
            }
            
        except Exception as e:
            print(f"计算CLIP分数时出错 ({video_path}): {e}")
            return None
    
    def calculate_temporal_consistency(self, video_path, max_frames=30, frame_interval=1):
        """
        计算视频的时间一致性（相邻帧之间的CLIP特征相似度）
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
        
        Returns:
            dict: 时间一致性分数
        """
        try:
            frames = self.extract_frames_from_video(video_path, max_frames, frame_interval)
            
            if len(frames) < 2:
                return {
                    'mean_temporal_score': None,
                    'std_temporal_score': None,
                    'temporal_scores': [],
                    'num_pairs': 0,
                    'error': 'Not enough frames for temporal analysis'
                }
            
            # 使用processor处理图像
            inputs = self.processor(
                images=frames,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 获取图像特征
                image_features = self.model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
                
                # 标准化特征
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # 计算相邻帧之间的相似度
                temporal_scores = []
                for i in range(len(image_features) - 1):
                    similarity = torch.dot(image_features[i], image_features[i+1])
                    temporal_scores.append(similarity.item())
            
            return {
                'mean_temporal_score': float(np.mean(temporal_scores)),
                'std_temporal_score': float(np.std(temporal_scores)),
                'temporal_scores': temporal_scores,
                'num_pairs': len(temporal_scores)
            }
            
        except Exception as e:
            print(f"计算时间一致性时出错 ({video_path}): {e}")
            return None
    
    def batch_calculate_clip_scores(self, video_text_pairs, max_frames=30, frame_interval=1):
        """
        批量计算多个视频-文本对的CLIP分数
        
        Args:
            video_text_pairs: [(video_path, text_prompt), ...] 视频文本对列表
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
        
        Returns:
            list: 结果列表
        """
        results = []
        
        for video_path, text_prompt in tqdm(video_text_pairs, desc="计算CLIP分数"):
            result = self.calculate_clip_score(video_path, text_prompt, max_frames, frame_interval)
            if result:
                results.append(result)
            else:
                print(f"跳过失败的评估: {video_path}")
        
        return results

def read_scenario_descriptions(description_file):
    """
    读取场景描述文件
    """
    scenarios = {}
    current_id = None
    
    try:
        with open(description_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.isdigit():
                current_id = line
                scenarios[current_id] = {}
            elif line.startswith('Safe:'):
                if current_id:
                    scenarios[current_id]['safe'] = line[5:].strip()
            elif line.startswith('Near-Crash:'):
                if current_id:
                    scenarios[current_id]['near-crash'] = line[11:].strip()
            elif line.startswith('Crash:'):
                if current_id:
                    scenarios[current_id]['crash'] = line[6:].strip()
        
        return scenarios
        
    except Exception as e:
        raise RuntimeError(f"读取描述文件失败: {e}")

def find_video_files(test_data_dir, scenario_id):
    """
    查找指定场景的视频文件
    """
    scenario_dir = None
    for candidate in normalize_scenario_id(scenario_id):
        candidate_dir = os.path.join(test_data_dir, candidate)
        if os.path.exists(candidate_dir):
            scenario_dir = candidate_dir
            break

    video_files = {}
    
    if not scenario_dir:
        return video_files
    
    # 支持更多视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for file in os.listdir(scenario_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            file_lower = file.lower()
            if 'safe' in file_lower:
                video_files['safe'] = os.path.join(scenario_dir, file)
            elif 'crash' in file_lower and 'near' not in file_lower:
                video_files['crash'] = os.path.join(scenario_dir, file)
            elif 'near' in file_lower or 'near-crash' in file_lower:
                video_files['near-crash'] = os.path.join(scenario_dir, file)
    
    return video_files

def evaluate_scenario(evaluator, scenario_id, test_data_dir, descriptions, 
                     max_frames=30, frame_interval=1):
    """
    评估单个场景的所有视频
    """
    scenario_descriptions = {}
    matched_scenario_id = None
    for candidate in normalize_scenario_id(scenario_id):
        if candidate in descriptions:
            scenario_descriptions = descriptions[candidate]
            matched_scenario_id = candidate
            break

    video_files = find_video_files(test_data_dir, scenario_id)
    
    if not scenario_descriptions:
        print(f"警告: 场景 {scenario_id} 缺少描述")
        return None
    
    if not video_files:
        print(f"警告: 场景 {scenario_id} 缺少视频文件")
        return None
    
    results = {
        'scenario_id': matched_scenario_id or scenario_id,
        'videos': {},
        'summary': {}
    }
    
    print(f"\n=== 评估场景 {scenario_id} ===")
    
    for video_type in ['safe', 'near-crash', 'crash']:
        video_path = video_files.get(video_type)
        text_prompt = scenario_descriptions.get(video_type)
        
        if not video_path or not text_prompt:
            print(f"跳过 {video_type}: 缺少文件或描述")
            continue
        
        print(f"处理 {video_type} 视频: {os.path.basename(video_path)}")
        
        # 计算CLIP分数
        clip_result = evaluator.calculate_clip_score(
            video_path, text_prompt, max_frames, frame_interval
        )
        
        # 计算时间一致性
        temporal_result = evaluator.calculate_temporal_consistency(
            video_path, max_frames, frame_interval
        )
        
        if clip_result:
            results['videos'][video_type] = {
                'video_path': video_path,
                'text_prompt': text_prompt,
                'clip_score': clip_result['mean_score'],
                'clip_std': clip_result['std_score'],
                'clip_max': clip_result['max_score'],
                'clip_min': clip_result['min_score'],
                'frame_scores': clip_result['frame_scores'],
                'num_frames': clip_result['num_frames'],
                'temporal_consistency': temporal_result['mean_temporal_score'] if temporal_result else None,
                'temporal_std': temporal_result['std_temporal_score'] if temporal_result else None
            }
            
            print(f"  CLIP分数: {clip_result['mean_score']:.4f} ± {clip_result['std_score']:.4f}")
            print(f"  CLIP范围: [{clip_result['min_score']:.4f}, {clip_result['max_score']:.4f}]")
            if temporal_result and temporal_result['mean_temporal_score'] is not None:
                print(f"  时间一致性: {temporal_result['mean_temporal_score']:.4f} ± {temporal_result['std_temporal_score']:.4f}")
        else:
            print(f"  {video_type} 视频评估失败")
    
    # 计算场景总结
    if results['videos']:
        clip_scores = [v['clip_score'] for v in results['videos'].values()]
        temporal_scores = [v['temporal_consistency'] for v in results['videos'].values() 
                          if v['temporal_consistency'] is not None]
        
        results['summary'] = {
            'avg_clip_score': float(np.mean(clip_scores)),
            'std_clip_score': float(np.std(clip_scores)),
            'max_clip_score': float(np.max(clip_scores)),
            'min_clip_score': float(np.min(clip_scores)),
            'avg_temporal_score': float(np.mean(temporal_scores)) if temporal_scores else None,
            'num_videos': len(results['videos'])
        }
        
        print(f"\n场景总结:")
        print(f"  平均CLIP分数: {results['summary']['avg_clip_score']:.4f} ± {results['summary']['std_clip_score']:.4f}")
        print(f"  CLIP分数范围: [{results['summary']['min_clip_score']:.4f}, {results['summary']['max_clip_score']:.4f}]")
        if results['summary']['avg_temporal_score']:
            print(f"  平均时间一致性: {results['summary']['avg_temporal_score']:.4f}")
    
    return results

def save_results_to_json(results, output_file="results/clip_evaluation_results.json"):
    """
    保存结果到JSON文件
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 添加元数据
    output_data = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'num_scenarios': len([r for r in results if r is not None]),
            'total_videos': sum(len(r['videos']) for r in results if r is not None)
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")


def save_results_to_csv(results, output_file="results/clip_evaluation_results.csv"):
    """
    保存扁平化结果到CSV文件
    """
    import csv

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fieldnames = [
        "scenario_id",
        "video_type",
        "clip_score_x100",
        "clip_std_x100",
        "clip_min_x100",
        "clip_max_x100",
        "temporal_consistency",
        "temporal_std",
        "num_frames",
        "video_path",
        "text_prompt",
    ]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if not result:
                continue
            for video_type, video_data in result["videos"].items():
                writer.writerow(
                    {
                        "scenario_id": result["scenario_id"],
                        "video_type": video_type,
                        "clip_score_x100": round(video_data["clip_score"] * 100, 3),
                        "clip_std_x100": round(video_data["clip_std"] * 100, 3),
                        "clip_min_x100": round(video_data["clip_min"] * 100, 3),
                        "clip_max_x100": round(video_data["clip_max"] * 100, 3),
                        "temporal_consistency": (
                            round(video_data["temporal_consistency"], 4)
                            if video_data["temporal_consistency"] is not None
                            else None
                        ),
                        "temporal_std": (
                            round(video_data["temporal_std"], 4)
                            if video_data["temporal_std"] is not None
                            else None
                        ),
                        "num_frames": video_data["num_frames"],
                        "video_path": video_data["video_path"],
                        "text_prompt": video_data["text_prompt"],
                    }
                )

    print(f"CSV结果已保存到: {output_file}")


def discover_scenario_ids(test_data_dir):
    """Discover numeric scenario folders directly from the dataset."""
    if not os.path.exists(test_data_dir):
        return []

    scenario_ids = []
    for entry in sorted(os.listdir(test_data_dir)):
        full_path = os.path.join(test_data_dir, entry)
        if os.path.isdir(full_path) and entry.isdigit():
            scenario_ids.append(entry)
    return scenario_ids

def main():
    parser = argparse.ArgumentParser(description='改进的视频CLIP相似度评估器')
    parser.add_argument('--test_data_dir', type=str, default='test_data',
                       help='测试数据目录路径')
    parser.add_argument('--description_file', type=str, default='test_data/description.txt',
                       help='场景描述文件路径')
    parser.add_argument('--scenario_ids', type=str, nargs='+', 
                       default=None,
                       help='要评估的场景ID列表')
    parser.add_argument('--max_frames', type=int, default=30,
                       help='每个视频最大处理帧数')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='帧间隔（每隔几帧取一帧）')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP模型名称')
    parser.add_argument('--local_files_only', action='store_true',
                       help='只使用本地缓存模型，不访问网络')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.output_dir, f"clip_evaluation_{timestamp}.log")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始改进的视频CLIP评估")
    logger.info(f"参数设置: {vars(args)}")
    
    try:
        # 初始化评估器
        evaluator = ImprovedVideoCLIPEvaluator(
            device=args.device,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
        )
        
        # 读取场景描述
        descriptions = read_scenario_descriptions(args.description_file)
        logger.info(f"成功加载 {len(descriptions)} 个场景的描述")

        scenario_ids = args.scenario_ids or discover_scenario_ids(args.test_data_dir)
        logger.info(f"待评估场景: {scenario_ids}")
        
        # 评估所有场景
        all_results = []
        successful_evaluations = 0
        
        for scenario_id in tqdm(scenario_ids, desc="评估场景"):
            try:
                result = evaluate_scenario(
                    evaluator, scenario_id, args.test_data_dir, descriptions,
                    args.max_frames, args.frame_interval
                )
                if result:
                    all_results.append(result)
                    successful_evaluations += 1
                    logger.info(f"场景 {scenario_id} 评估完成")
                else:
                    all_results.append(None)
                    logger.warning(f"场景 {scenario_id} 评估失败")
            except Exception as e:
                logger.error(f"场景 {scenario_id} 评估出错: {e}")
                all_results.append(None)
        
        # 保存结果
        save_results_to_json(all_results, os.path.join(args.output_dir, 'clip_evaluation_results.json'))
        save_results_to_csv(all_results, os.path.join(args.output_dir, 'clip_evaluation_results.csv'))
        
        # 打印总结
        logger.info("\n=== 评估总结 ===")
        logger.info(f"成功评估场景数: {successful_evaluations}/{len(scenario_ids)}")
        
        valid_results = [r for r in all_results if r is not None]
        if valid_results:
            total_videos = sum(len(r['videos']) for r in valid_results)
            all_clip_scores = []
            all_temporal_scores = []
            for result in valid_results:
                for video_data in result['videos'].values():
                    all_clip_scores.append(video_data['clip_score'])
                    if video_data['temporal_consistency'] is not None:
                        all_temporal_scores.append(video_data['temporal_consistency'])
            
            if all_clip_scores:
                logger.info(f"总评估视频数: {total_videos}")
                logger.info(f"平均CLIP分数: {np.mean(all_clip_scores):.4f} ± {np.std(all_clip_scores):.4f}")
                logger.info(f"CLIP分数范围: [{np.min(all_clip_scores):.4f}, {np.max(all_clip_scores):.4f}]")
            if all_temporal_scores:
                logger.info(f"平均时间一致性: {np.mean(all_temporal_scores):.4f} ± {np.std(all_temporal_scores):.4f}")
        
        logger.info("评估完成!")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()
