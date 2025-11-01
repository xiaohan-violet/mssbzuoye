import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import os
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import DEVICE
from utils.data_loader import get_data_loaders
from models.text_models import TextModel, TextCNNModel
from models.audio_models import AudioFCN, AudioSimple
from models.video_models import VideoModel
from models.fusion_models import MTFN, MLMF, LateFusion


class ComprehensiveEvaluator:
    """全方位模型评估器"""

    def __init__(self, model, model_name, test_loader, class_names=['负面', '中性', '正面']):
        self.model = model.to(DEVICE)
        self.model_name = model_name
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = len(class_names)

        # 加载最佳模型
        model_path = f'checkpoints/best_{model_name}.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"已加载最佳模型: {model_path}")
        else:
            print(f"警告: 未找到模型文件 {model_path}")
            return

        # 创建结果目录
        self.results_dir = f'results/{model_name}'
        os.makedirs(self.results_dir, exist_ok=True)

        # 存储预测结果
        self.all_predictions = None
        self.all_labels = None
        self.all_probabilities = None

    def evaluate(self):
        """执行完整评估"""
        print(f"\n开始评估 {self.model_name}...")

        # 收集预测结果
        self._collect_predictions()

        # 计算各种指标
        results = self._calculate_metrics()

        # 生成可视化图表
        self._generate_visualizations(results)

        # 保存详细结果
        self._save_detailed_results(results)

        return results

    def _collect_predictions(self):
        """收集模型预测结果"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f'预测 {self.model_name}'):
                if batch is None:
                    continue

                try:
                    inputs, labels = self._prepare_batch(batch)
                    labels = labels.to(DEVICE)

                    # 前向传播
                    if self.model_name in ['text_model', 'text_cnn']:
                        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                    elif self.model_name in ['audio_fcn', 'audio_simple']:
                        outputs = self.model(inputs.float())
                    elif self.model_name in ['video_model']:
                        outputs = self.model(inputs.float())
                    elif self.model_name in ['mtfn', 'mlmf', 'late_fusion']:
                        outputs = self.model(
                            inputs['text'].float(),
                            inputs['audio'].float(),
                            inputs['video'].float()
                        )

                    # 获取概率和预测
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

                except Exception as e:
                    print(f"预测批次错误: {e}")
                    continue

        self.all_predictions = np.array(all_preds)
        self.all_labels = np.array(all_labels)
        self.all_probabilities = np.array(all_probs)

    def _prepare_batch(self, batch):
        """准备批次数据"""
        labels = batch['label']

        if self.model_name in ['text_model', 'text_cnn']:
            inputs = {
                'input_ids': batch['text_input']['input_ids'].to(DEVICE),
                'attention_mask': batch['text_input']['attention_mask'].to(DEVICE)
            }
        elif self.model_name in ['audio_fcn', 'audio_simple']:
            inputs = batch['audio'].to(DEVICE)
        elif self.model_name in ['video_model']:
            inputs = batch['video'].to(DEVICE)
        elif self.model_name in ['mtfn', 'mlmf', 'late_fusion']:
            inputs = {
                'text': batch['text_features'].to(DEVICE),
                'audio': batch['audio'].to(DEVICE),
                'video': batch['video'].to(DEVICE)
            }

        return inputs, labels

    def _calculate_metrics(self):
        """计算各种评估指标"""
        if self.all_predictions is None:
            return {}

        # 基础指标
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        precision = precision_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)
        recall = recall_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)

        # 宏平均指标
        precision_macro = precision_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)

        # 分类报告
        class_report = classification_report(
            self.all_labels, self.all_predictions,
            target_names=self.class_names, output_dict=True
        )

        # 混淆矩阵
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        # AUC-ROC 指标（多分类）
        try:
            # 将标签二值化用于多分类ROC
            y_true_bin = label_binarize(self.all_labels, classes=[0, 1, 2])
            auc_roc = roc_auc_score(y_true_bin, self.all_probabilities, average='weighted', multi_class='ovr')
        except:
            auc_roc = 0.0

        # 平均精度 (Average Precision)
        try:
            avg_precision = average_precision_score(y_true_bin, self.all_probabilities, average='weighted')
        except:
            avg_precision = 0.0

        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc_roc': auc_roc,
            'average_precision': avg_precision,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': self.all_predictions.tolist(),
            'labels': self.all_labels.tolist(),
            'probabilities': self.all_probabilities.tolist()
        }

        return results

    def _generate_visualizations(self, results):
        """生成各种可视化图表"""
        if not results:
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 混淆矩阵热力图
        self._plot_confusion_matrix(results['confusion_matrix'])

        # 2. ROC曲线
        self._plot_roc_curve()

        # 3. 精确率-召回率曲线
        self._plot_precision_recall_curve()

        # 4. 类别分布图
        self._plot_class_distribution()

        # 5. 模型性能对比雷达图（单个模型）
        self._plot_performance_radar(results)

        # 6. 预测概率分布图
        self._plot_probability_distribution()

    def _plot_confusion_matrix(self, cm):
        """绘制混淆矩阵热力图"""
        plt.figure(figsize=(10, 8))
        cm_array = np.array(cm)

        # 计算百分比
        cm_percent = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': '百分比 (%)'})

        plt.title(f'{self.model_name} - 混淆矩阵 (百分比)')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrix.pdf', bbox_inches='tight')
        plt.close()

        # 同时保存原始数值的混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{self.model_name} - 混淆矩阵 (样本数)')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrix_counts.pdf', bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(self):
        """绘制ROC曲线"""
        y_true_bin = label_binarize(self.all_labels, classes=[0, 1, 2])

        plt.figure(figsize=(10, 8))

        # 为每个类别绘制ROC曲线
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.all_probabilities[:, i])
            auc = roc_auc_score(y_true_bin[:, i], self.all_probabilities[:, i])
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.3f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title(f'{self.model_name} - ROC曲线')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/roc_curve.pdf', bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curve(self):
        """绘制精确率-召回率曲线"""
        y_true_bin = label_binarize(self.all_labels, classes=[0, 1, 2])

        plt.figure(figsize=(10, 8))

        # 为每个类别绘制PR曲线
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], self.all_probabilities[:, i])
            ap = average_precision_score(y_true_bin[:, i], self.all_probabilities[:, i])
            plt.plot(recall, precision, label=f'{self.class_names[i]} (AP = {ap:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'{self.model_name} - 精确率-召回率曲线')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/precision_recall_curve.pdf', bbox_inches='tight')
        plt.close()

    def _plot_class_distribution(self):
        """绘制类别分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 真实标签分布
        true_counts = [np.sum(self.all_labels == i) for i in range(self.num_classes)]
        ax1.bar(self.class_names, true_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax1.set_title('真实标签分布')
        ax1.set_ylabel('样本数量')

        # 预测标签分布
        pred_counts = [np.sum(self.all_predictions == i) for i in range(self.num_classes)]
        ax2.bar(self.class_names, pred_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_title('预测标签分布')
        ax2.set_ylabel('样本数量')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/class_distribution.pdf', bbox_inches='tight')
        plt.close()

    def _plot_performance_radar(self, results):
        """绘制性能雷达图"""
        # 选择关键指标
        metrics = ['准确率', '加权F1', '宏平均F1', 'AUC-ROC', '平均精度']
        values = [
            results['accuracy'],
            results['f1_weighted'],
            results['f1_macro'],
            results['auc_roc'],
            results['average_precision']
        ]

        # 雷达图需要闭合
        metrics_radar = metrics + [metrics[0]]
        values_radar = values + [values[0]]

        # 创建角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values_radar, 'o-', linewidth=2, label=self.model_name)
        ax.fill(angles, values_radar, alpha=0.25)

        # 修复：确保刻度和标签数量一致
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'{self.model_name} - 性能雷达图', size=16, y=1.08)
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_radar.pdf', bbox_inches='tight')
        plt.close()

    def _plot_probability_distribution(self):
        """绘制预测概率分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, class_name in enumerate(self.class_names):
            # 获取该类别的预测概率
            class_probs = self.all_probabilities[:, i]

            # 根据真实标签分类
            true_positive_probs = class_probs[self.all_labels == i]
            false_positive_probs = class_probs[(self.all_labels != i) & (self.all_predictions == i)]

            axes[i].hist(true_positive_probs, bins=20, alpha=0.7, label='真阳性', color='green')
            axes[i].hist(false_positive_probs, bins=20, alpha=0.7, label='假阳性', color='red')

            axes[i].set_xlabel('预测概率')
            axes[i].set_ylabel('频次')
            axes[i].set_title(f'{class_name} - 概率分布')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/probability_distribution.pdf', bbox_inches='tight')
        plt.close()

    def _save_detailed_results(self, results):
        """保存详细结果"""
        # 保存JSON格式的详细结果
        with open(f'{self.results_dir}/detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存文本格式的摘要报告
        with open(f'{self.results_dir}/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== {self.model_name} 评估报告 ===\n\n")
            f.write("主要指标:\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"加权F1分数: {results['f1_weighted']:.4f}\n")
            f.write(f"宏平均F1分数: {results['f1_macro']:.4f}\n")
            f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n")
            f.write(f"平均精度: {results['average_precision']:.4f}\n\n")

            f.write("分类报告:\n")
            class_report = results['classification_report']
            for class_name in self.class_names:
                f.write(f"{class_name}:\n")
                f.write(f"  精确率: {class_report[class_name]['precision']:.4f}\n")
                f.write(f"  召回率: {class_report[class_name]['recall']:.4f}\n")
                f.write(f"  F1分数: {class_report[class_name]['f1-score']:.4f}\n")
                f.write(f"  支持数: {class_report[class_name]['support']}\n\n")


def evaluate_all_models():
    """评估所有模型"""
    print("开始全方位评估所有模型...")

    # 确保主结果目录存在
    os.makedirs('results', exist_ok=True)

    all_results = {}
    models_to_evaluate = [
        ('text_model', TextModel, 'text'),
        ('text_cnn', TextCNNModel, 'text'),
        ('audio_fcn', AudioFCN, 'audio'),
        ('audio_simple', AudioSimple, 'audio'),
        ('video_model', VideoModel, 'video'),
        ('mtfn', MTFN, 'all'),
        ('mlmf', MLMF, 'all'),
        ('late_fusion', LateFusion, 'all')
    ]

    for model_name, model_class, modality in models_to_evaluate:
        # 检查模型文件是否存在
        model_path = f'checkpoints/best_{model_name}.pth'
        if not os.path.exists(model_path):
            print(f"跳过 {model_name} - 模型文件不存在")
            continue

        # 获取数据加载器
        _, _, test_loader = get_data_loaders(batch_size=32, modality=modality)

        # 创建模型实例
        model = model_class()

        # 评估模型
        evaluator = ComprehensiveEvaluator(model, model_name, test_loader)
        results = evaluator.evaluate()

        if results:
            all_results[model_name] = results
            print(f"{model_name} 评估完成")

    # 生成模型比较报告
    if all_results:
        generate_comparison_report(all_results)

    return all_results


def generate_comparison_report(all_results):
    """生成模型比较报告"""
    print("\n生成模型比较报告...")

    # 创建比较目录
    comp_dir = 'results/comparison'
    os.makedirs(comp_dir, exist_ok=True)

    # 提取关键指标
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            '模型': model_name,
            '准确率': results['accuracy'],
            '加权F1': results['f1_weighted'],
            '宏平均F1': results['f1_macro'],
            'AUC-ROC': results['auc_roc'],
            '平均精度': results['average_precision']
        })

    # 创建比较DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('准确率', ascending=False)

    # 保存比较表格
    df_comparison.to_csv(f'{comp_dir}/model_comparison.csv', index=False, encoding='utf-8-sig')
    df_comparison.to_excel(f'{comp_dir}/model_comparison.xlsx', index=False)

    # 生成比较图表
    _plot_model_comparison(df_comparison, comp_dir)

    # 保存详细比较报告
    with open(f'{comp_dir}/detailed_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("=== 模型性能比较报告 ===\n\n")
        f.write("性能排名:\n")
        for i, (_, row) in enumerate(df_comparison.iterrows(), 1):
            f.write(f"{i}. {row['模型']}: 准确率={row['准确率']:.4f}, F1加权={row['加权F1']:.4f}\n")

        f.write(f"\n最佳模型: {df_comparison.iloc[0]['模型']}\n")
        f.write(f"最佳准确率: {df_comparison.iloc[0]['准确率']:.4f}\n")
        f.write(f"最佳F1分数: {df_comparison.iloc[0]['加权F1']:.4f}\n")

    print(f"比较报告已保存到 {comp_dir}/")


def _plot_model_comparison(df, save_dir):
    """绘制模型比较图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 条形图比较
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['准确率', '加权F1', '宏平均F1', 'AUC-ROC']
    df_columns = ['准确率', '加权F1', '宏平均F1', 'AUC-ROC']

    for i, (metric, col) in enumerate(zip(metrics, df_columns)):
        ax = axes[i // 2, i % 2]
        bars = ax.barh(df['模型'], df[col], color=plt.cm.Set3(range(len(df))))
        ax.set_xlabel(metric)
        ax.set_title(f'模型{metric}比较')

        # 在条形上添加数值
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_bars.pdf', bbox_inches='tight')
    plt.close()

    # 2. 热力图比较
    plt.figure(figsize=(12, 8))
    comparison_metrics = df[['准确率', '加权F1', '宏平均F1', 'AUC-ROC', '平均精度']]
    sns.heatmap(comparison_metrics.set_index(df['模型']),
                annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': '分数'})
    plt.title('模型性能热力图比较')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_heatmap.pdf', bbox_inches='tight')
    plt.close()

    # 3. 雷达图比较（前3个模型）
    if len(df) >= 3:
        top_models = df.head(3)
        metrics_radar = ['准确率', '加权F1', '宏平均F1', 'AUC-ROC', '平均精度']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 为每个模型绘制雷达图
        for idx, (_, row) in enumerate(top_models.iterrows()):
            values = [row[metric] for metric in metrics_radar]

            # 雷达图需要闭合
            values_radar = values + [values[0]]
            metrics_radar_closed = metrics_radar + [metrics_radar[0]]

            # 创建角度
            angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            ax.plot(angles, values_radar, 'o-', linewidth=2, label=row['模型'])
            ax.fill(angles, values_radar, alpha=0.1)

        # 修复：确保刻度和标签数量一致
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_radar)
        ax.set_ylim(0, 1)
        ax.set_title('Top 3 模型性能雷达图比较', size=16, y=1.08)
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/top3_models_radar.pdf', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # 评估所有模型
    results = evaluate_all_models()

    print("\n" + "=" * 60)
    print("全方位评估完成！")
    print("=" * 60)
    print("生成的图表和报告保存在 'results/' 目录下")
    print("每个模型都有独立的评估文件夹")
    print("模型比较报告保存在 'results/comparison/' 目录下")