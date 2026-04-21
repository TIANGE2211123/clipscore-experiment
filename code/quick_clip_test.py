#!/usr/bin/env python3
"""
改进的快速CLIP测试脚本
用于快速测试单个场景或少量视频的CLIP相似度
适配改进的VideoCLIPEvaluator
"""

import os
import sys
import time
import argparse
from pathlib import Path
import traceback

# 导入改进的评估器
try:
    from video_clip_evaluator import (
        ImprovedVideoCLIPEvaluator,
        read_scenario_descriptions,
        find_video_files,
        normalize_scenario_id,
    )
except ImportError:
    print("错误: 无法导入 ImprovedVideoCLIPEvaluator")
    print("请确保 improved_video_clip_evaluator.py 文件在同一目录下")
    sys.exit(1)

def resolve_description_key(descriptions, scenario_id):
    """Support both 1376 and 001376 style IDs."""
    for candidate in normalize_scenario_id(scenario_id):
        if candidate in descriptions:
            return candidate
    return None


class QuickTester:
    """快速测试器类"""
    
    def __init__(self, device=None, model_name="openai/clip-vit-base-patch32", local_files_only=False):
        """
        初始化快速测试器
        
        Args:
            device: 计算设备
            model_name: CLIP模型名称
        """
        self.device = device
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.evaluator = None
        self._init_evaluator()
    
    def _init_evaluator(self):
        """初始化评估器（懒加载）"""
        try:
            print("正在初始化CLIP评估器...")
            start_time = time.time()
            self.evaluator = ImprovedVideoCLIPEvaluator(
                device=self.device,
                model_name=self.model_name,
                local_files_only=self.local_files_only,
            )
            init_time = time.time() - start_time
            print(f"评估器初始化完成 (耗时: {init_time:.2f}秒)")
        except Exception as e:
            print(f"评估器初始化失败: {e}")
            raise
    
    def test_scenario(self, scenario_id="1376", test_data_dir="test_data", max_frames=30, frame_interval=1):
        """
        测试单个场景
        
        Args:
            scenario_id: 场景ID
            test_data_dir: 测试数据目录
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
        """
        print(f"\n{'='*60}")
        print(f"快速测试场景 {scenario_id}")
        print(f"{'='*60}")
        
        try:
            # 读取场景描述
            description_file = os.path.join(test_data_dir, "description.txt")
            if not os.path.exists(description_file):
                print(f"错误: 描述文件不存在 {description_file}")
                return False
            
            descriptions = read_scenario_descriptions(description_file)
            
            resolved_id = resolve_description_key(descriptions, scenario_id)
            if not resolved_id:
                print(f"错误: 找不到场景 {scenario_id} 的描述")
                print(f"可用场景ID: {list(descriptions.keys())}")
                return False
            
            # 查找视频文件
            video_files = find_video_files(test_data_dir, scenario_id)
            
            if not video_files:
                print(f"错误: 找不到场景 {scenario_id} 的视频文件")
                print(f"查找目录: {os.path.join(test_data_dir, scenario_id)}")
                return False
            
            # 显示场景信息
            print(f"场景描述:")
            for desc_type, desc_text in descriptions[resolved_id].items():
                print(f"  {desc_type}: {desc_text}")
            
            print(f"\n找到的视频文件:")
            for video_type, video_path in video_files.items():
                print(f"  {video_type}: {os.path.basename(video_path)}")
            
            print(f"\n处理参数: 最大帧数={max_frames}, 帧间隔={frame_interval}")
            print("-" * 60)
            
            # 评估每个视频
            results = {}
            total_start_time = time.time()
            
            for video_type, video_path in video_files.items():
                text_prompt = descriptions[resolved_id].get(video_type)
                
                if not text_prompt:
                    print(f"跳过 {video_type}: 缺少文本描述")
                    continue
                
                print(f"\n评估 {video_type.upper()} 视频")
                print(f"   视频: {os.path.basename(video_path)}")
                print(f"   提示: {text_prompt}")
                
                # 计算CLIP分数
                start_time = time.time()
                clip_result = self.evaluator.calculate_clip_score(
                    video_path, text_prompt, max_frames, frame_interval
                )
                clip_time = time.time() - start_time
                
                if clip_result:
                    results[video_type] = clip_result
                    
                    print(f"   CLIP分数: {clip_result['mean_score']:.4f} ± {clip_result['std_score']:.4f}")
                    print(f"   分数范围: [{clip_result['min_score']:.4f}, {clip_result['max_score']:.4f}]")
                    print(f"   处理帧数: {clip_result['num_frames']}")
                    print(f"   处理时间: {clip_time:.2f}秒")
                    
                    # 计算时间一致性
                    temporal_result = self.evaluator.calculate_temporal_consistency(
                        video_path, max_frames, frame_interval
                    )
                    if temporal_result and temporal_result['mean_temporal_score'] is not None:
                        print(f"   时间一致性: {temporal_result['mean_temporal_score']:.4f} ± {temporal_result['std_temporal_score']:.4f}")
                    
                    # 显示部分逐帧分数
                    frame_scores = clip_result['frame_scores']
                    if len(frame_scores) > 5:
                        sample_scores = frame_scores[:5] + ["..."] + frame_scores[-2:]
                    else:
                        sample_scores = frame_scores
                    print(f"   帧分数样本: {[f'{s:.3f}' if isinstance(s, (int, float)) else s for s in sample_scores]}")
                    
                else:
                    print(f"   评估失败")
                
                print("-" * 40)
            
            total_time = time.time() - total_start_time
            
            # 打印总结
            if results:
                self._print_scenario_summary(results, scenario_id, total_time)
                return True
            else:
                print("没有成功评估的视频")
                return False
                
        except Exception as e:
            print(f"场景测试出错: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
            return False
    
    def test_single_video(self, video_path, text_prompt, max_frames=73, frame_interval=1):
        """
        测试单个视频
        
        Args:
            video_path: 视频文件路径
            text_prompt: 文本提示
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
        """
        print(f"\n{'='*60}")
        print(f"快速测试单个视频")
        print(f"{'='*60}")
        
        print(f"视频路径: {video_path}")
        print(f"文本提示: {text_prompt}")
        print(f"参数: 最大帧数={max_frames}, 帧间隔={frame_interval}")
        
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 {video_path}")
            return False
        
        try:
            print(f"\n{'开始评估':<20}")
            total_start_time = time.time()
            
            # 计算CLIP分数
            print("计算CLIP相似度...")
            start_time = time.time()
            clip_result = self.evaluator.calculate_clip_score(
                video_path, text_prompt, max_frames, frame_interval
            )
            clip_time = time.time() - start_time
            
            if clip_result:
                print(f"   CLIP分数: {clip_result['mean_score']:.4f} ± {clip_result['std_score']:.4f}")
                print(f"   分数范围: [{clip_result['min_score']:.4f}, {clip_result['max_score']:.4f}]")
                print(f"   处理帧数: {clip_result['num_frames']}")
                print(f"   计算时间: {clip_time:.2f}秒")
                
                # 计算时间一致性
                print("\n计算时间一致性...")
                start_time = time.time()
                temporal_result = self.evaluator.calculate_temporal_consistency(
                    video_path, max_frames, frame_interval
                )
                temporal_time = time.time() - start_time
                
                if temporal_result and temporal_result['mean_temporal_score'] is not None:
                    print(f"   时间一致性: {temporal_result['mean_temporal_score']:.4f} ± {temporal_result['std_temporal_score']:.4f}")
                    print(f"   计算时间: {temporal_time:.2f}秒")
                else:
                    print(f"   时间一致性计算失败")
                
                total_time = time.time() - total_start_time
                
                # 显示详细分析
                print(f"\n{'详细分析':<20}")
                print("-" * 40)
                
                # 逐帧分数分析
                frame_scores = clip_result['frame_scores']
                print(f"逐帧分数统计:")
                print(f"   最高分: {max(frame_scores):.4f} (第{frame_scores.index(max(frame_scores))+1}帧)")
                print(f"   最低分: {min(frame_scores):.4f} (第{frame_scores.index(min(frame_scores))+1}帧)")
                print(f"   分数方差: {clip_result['std_score']:.4f}")
                
                # 分数分布
                high_scores = [s for s in frame_scores if s > clip_result['mean_score'] + clip_result['std_score']]
                low_scores = [s for s in frame_scores if s < clip_result['mean_score'] - clip_result['std_score']]
                print(f"   高分帧数: {len(high_scores)}/{len(frame_scores)} ({len(high_scores)/len(frame_scores)*100:.1f}%)")
                print(f"   低分帧数: {len(low_scores)}/{len(frame_scores)} ({len(low_scores)/len(frame_scores)*100:.1f}%)")
                
                # 质量评估
                print(f"\n质量评估:")
                self._analyze_video_quality(clip_result, temporal_result)
                
                print(f"\n总耗时: {total_time:.2f}秒")
                return True
                
            else:
                print("CLIP分数计算失败")
                return False
                
        except Exception as e:
            print(f"单个视频测试出错: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
            return False
    
    def _print_scenario_summary(self, results, scenario_id, total_time):
        """打印场景总结"""
        print(f"\n{'场景总结':<20}")
        print("=" * 60)
        
        clip_scores = [r['mean_score'] for r in results.values()]
        video_types = list(results.keys())
        
        print(f"整体统计:")
        print(f"   场景ID: {scenario_id}")
        print(f"   评估视频数: {len(results)}")
        print(f"   平均CLIP分数: {sum(clip_scores)/len(clip_scores):.4f}")
        print(f"   分数标准差: {(sum([(s - sum(clip_scores)/len(clip_scores))**2 for s in clip_scores])/len(clip_scores))**0.5:.4f}")
        
        # 找出最高和最低分数
        max_score = max(clip_scores)
        min_score = min(clip_scores)
        max_idx = clip_scores.index(max_score)
        min_idx = clip_scores.index(min_score)
        
        print(f"   最高分: {max_score:.4f} ({video_types[max_idx]})")
        print(f"   最低分: {min_score:.4f} ({video_types[min_idx]})")
        print(f"   分数差: {max_score - min_score:.4f}")
        
        print(f"\n各视频类型排名:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        for i, (video_type, result) in enumerate(sorted_results, 1):
            print(f"   {i}. {video_type.upper()}: {result['mean_score']:.4f}")
        
        # 分析建议
        print(f"\n分析建议:")
        score_diff = max_score - min_score
        avg_score = sum(clip_scores) / len(clip_scores)
        
        if score_diff > 0.15:
            print("   不同场景类型的CLIP相似度差异较大，模型区分度良好")
        elif score_diff > 0.08:
            print("   不同场景类型的CLIP相似度差异中等，可考虑优化文本描述")
        else:
            print("   不同场景类型的CLIP相似度差异较小，建议检查文本提示区分度")
        
        if avg_score > 0.35:
            print("   整体CLIP相似度较高，视频与文本描述匹配良好")
        elif avg_score > 0.25:
            print("   整体CLIP相似度中等，有提升空间")
        else:
            print("   整体CLIP相似度较低，建议改进生成模型或文本提示")
        
        print(f"\n总处理时间: {total_time:.2f}秒")
    
    def _analyze_video_quality(self, clip_result, temporal_result):
        """分析视频质量"""
        mean_score = clip_result['mean_score']
        std_score = clip_result['std_score']
        
        # CLIP分数分析
        if mean_score > 0.4:
            print("   CLIP相似度优秀，视频内容与文本高度匹配")
        elif mean_score > 0.3:
            print("   CLIP相似度良好，视频内容与文本较好匹配")
        elif mean_score > 0.2:
            print("   CLIP相似度一般，视频内容与文本部分匹配")
        else:
            print("   CLIP相似度较低，视频内容与文本匹配度差")
        
        # 一致性分析
        if std_score < 0.05:
            print("   帧间一致性很高，视频内容稳定")
        elif std_score < 0.1:
            print("   帧间一致性较高，视频内容相对稳定")
        elif std_score < 0.15:
            print("   帧间一致性中等，视频内容有一定波动")
        else:
            print("   帧间一致性较低，视频内容波动较大")
        
        # 时间一致性分析
        if temporal_result and temporal_result['mean_temporal_score'] is not None:
            temporal_score = temporal_result['mean_temporal_score']
            if temporal_score > 0.9:
                print("   时间连贯性优秀，相邻帧变化自然")
            elif temporal_score > 0.8:
                print("   时间连贯性良好，相邻帧变化较为自然")
            elif temporal_score > 0.7:
                print("   时间连贯性一般，相邻帧变化有些突兀")
            else:
                print("   时间连贯性较差，相邻帧变化明显")

def interactive_mode():
    """交互模式"""
    print("视频CLIP相似度快速测试工具")
    print("=" * 50)
    
    # 选择设备和模型
    device = None
    model_name = "openai/clip-vit-base-patch32"
    
    print("\n配置选项:")
    device_choice = input("选择计算设备 (1:自动, 2:CPU, 3:CUDA) [1]: ").strip() or "1"
    if device_choice == "2":
        device = "cpu"
    elif device_choice == "3":
        device = "cuda"
    
    model_choice = input("使用默认CLIP模型? (y/n) [y]: ").strip().lower() or "y"
    if model_choice == "n":
        custom_model = input("请输入模型名称: ").strip()
        if custom_model:
            model_name = custom_model
    
    try:
        tester = QuickTester(device=device, model_name=model_name)
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("请选择测试模式:")
        print("1. 测试单个场景 (如 1376)")
        print("2. 测试单个视频文件")
        print("3. 列出可用场景")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            scenario_id = input("请输入场景ID [1376]: ").strip() or "1376"
            test_data_dir = input("请输入测试数据目录 [test_data]: ").strip() or "test_data"
            max_frames = input("请输入最大帧数 [30]: ").strip()
            max_frames = int(max_frames) if max_frames.isdigit() else 30
            frame_interval = input("请输入帧间隔 [1]: ").strip()
            frame_interval = int(frame_interval) if frame_interval.isdigit() else 1
            
            tester.test_scenario(scenario_id, test_data_dir, max_frames, frame_interval)
        
        elif choice == '2':
            video_path = input("请输入视频文件路径: ").strip()
            text_prompt = input("请输入文本提示: ").strip()
            
            if video_path and text_prompt:
                max_frames = input("请输入最大帧数 [30]: ").strip()
                max_frames = int(max_frames) if max_frames.isdigit() else 30
                frame_interval = input("请输入帧间隔 [1]: ").strip()
                frame_interval = int(frame_interval) if frame_interval.isdigit() else 1
                
                tester.test_single_video(video_path, text_prompt, max_frames, frame_interval)
            else:
                print("错误: 请提供视频路径和文本提示")
        
        elif choice == '3':
            test_data_dir = input("请输入测试数据目录 [test_data]: ").strip() or "test_data"
            list_available_scenarios(test_data_dir)
        
        elif choice == '4':
            print("再见!")
            break
        
        else:
            print("无效选择，请重新输入")

def list_available_scenarios(test_data_dir="test_data"):
    """列出可用场景"""
    print(f"\n📁 查找目录: {test_data_dir}")
    
    try:
        # 读取描述文件
        description_file = os.path.join(test_data_dir, "description.txt")
        if os.path.exists(description_file):
            descriptions = read_scenario_descriptions(description_file)
            print(f"描述文件中的场景 ({len(descriptions)}个):")
            for scenario_id in sorted(descriptions.keys()):
                print(f"   {scenario_id}")
        
        # 查找实际目录
        if os.path.exists(test_data_dir):
            dirs = [d for d in os.listdir(test_data_dir) 
                   if os.path.isdir(os.path.join(test_data_dir, d)) and d.isdigit()]
            print(f"实际场景目录 ({len(dirs)}个):")
            for dir_name in sorted(dirs):
                video_files = find_video_files(test_data_dir, dir_name)
                print(f"   {dir_name}: {list(video_files.keys()) if video_files else '无视频文件'}")
        else:
            print(f"目录不存在: {test_data_dir}")
            
    except Exception as e:
        print(f"列出场景时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进的快速CLIP测试脚本')
    parser.add_argument('mode', nargs='?', choices=['scenario', 'video', 'list'], 
                       help='测试模式: scenario/video/list')
    parser.add_argument('--scenario-id', type=str, default='001376',
                       help='场景ID (用于scenario模式)')
    parser.add_argument('--video-path', type=str,
                       help='视频文件路径 (用于video模式)')
    parser.add_argument('--text-prompt', type=str,
                       help='文本提示 (用于video模式)')
    parser.add_argument('--test-data-dir', type=str, default='test_data',
                       help='测试数据目录')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='最大处理帧数')
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='帧间隔')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                       help='计算设备')
    parser.add_argument('--model-name', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP模型名称')
    parser.add_argument('--local-files-only', action='store_true',
                       help='只使用本地缓存模型，不访问网络')
    
    args = parser.parse_args()
    
    # 如果没有提供模式，进入交互模式
    if not args.mode:
        interactive_mode()
        return
    
    try:
        if args.mode == 'list':
            list_available_scenarios(args.test_data_dir)
            return
        
        # 初始化测试器
        tester = QuickTester(
            device=args.device,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
        )
        
        if args.mode == 'scenario':
            success = tester.test_scenario(
                args.scenario_id, args.test_data_dir, 
                args.max_frames, args.frame_interval
            )
            sys.exit(0 if success else 1)
        
        elif args.mode == 'video':
            if not args.video_path or not args.text_prompt:
                print("错误: video模式需要提供 --video-path 和 --text-prompt")
                sys.exit(1)
            
            success = tester.test_single_video(
                args.video_path, args.text_prompt,
                args.max_frames, args.frame_interval
            )
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        sys.exit(130)
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
