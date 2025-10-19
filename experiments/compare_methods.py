"""对比不同方法的性能"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from visualization.visualizer import Visualizer


def generate_comparison_table():
    """
    生成论文中的表2-3：人工方案和DRL-TSBC的对比
    """
    print("生成性能对比表格...")
    
    # 论文中的数据
    results_208 = [
        {
            '方法': '人工方案',
            '方向': '上行',
            '发车次数': 72,
            '乘客平均等待时间(分钟)': 4.06,
            '被滞留乘客数量': 3,
            'ω': '-'
        },
        {
            '方法': '人工方案',
            '方向': '下行',
            '发车次数': 72,
            '乘客平均等待时间(分钟)': 4.6,
            '被滞留乘客数量': 176,
            'ω': '-'
        },
        {
            '方法': 'DRL-TO',
            '方向': '上行',
            '发车次数': 69,
            '乘客平均等待时间(分钟)': 3.7,
            '被滞留乘客数量': 0,
            'ω': '1/1000'
        },
        {
            '方法': 'DRL-TO',
            '方向': '下行',
            '发车次数': 72,
            '乘客平均等待时间(分钟)': 3.2,
            '被滞留乘客数量': 7,
            'ω': '1/500'
        },
        {
            '方法': 'DRL-TSBC',
            '方向': '上行',
            '发车次数': 73,
            '乘客平均等待时间(分钟)': 3.7,
            '被滞留乘客数量': 0,
            'ω': '1/1000'
        },
        {
            '方法': 'DRL-TSBC',
            '方向': '下行',
            '发车次数': 73,
            '乘客平均等待时间(分钟)': 3.8,
            '被滞留乘客数量': 0,
            'ω': '1/1000'
        }
    ]
    
    results_211 = [
        {
            '方法': '人工方案',
            '方向': '上行',
            '发车次数': 76,
            '乘客平均等待时间(分钟)': 4.68,
            '被滞留乘客数量': 0,
            'ω': '-'
        },
        {
            '方法': '人工方案',
            '方向': '下行',
            '发车次数': 76,
            '乘客平均等待时间(分钟)': 3.5,
            '被滞留乘客数量': 0,
            'ω': '-'
        },
        {
            '方法': 'DRL-TO',
            '方向': '上行',
            '发车次数': 75,
            '乘客平均等待时间(分钟)': 3.6,
            '被滞留乘客数量': 0,
            'ω': '1/500'
        },
        {
            '方法': 'DRL-TO',
            '方向': '下行',
            '发车次数': 76,
            '乘客平均等待时间(分钟)': 3.1,
            '被滞留乘客数量': 0,
            'ω': '1/300'
        },
        {
            '方法': 'DRL-TSBC',
            '方向': '上行',
            '发车次数': 75,
            '乘客平均等待时间(分钟)': 4.0,
            '被滞留乘客数量': 0,
            'ω': '1/900'
        },
        {
            '方法': 'DRL-TSBC',
            '方向': '下行',
            '发车次数': 75,
            '乘客平均等待时间(分钟)': 3.3,
            '被滞留乘客数量': 0,
            'ω': '1/900'
        }
    ]
    
    # 创建DataFrame
    df_208 = pd.DataFrame(results_208)
    df_211 = pd.DataFrame(results_211)
    
    # 保存表格
    vis = Visualizer()
    
    print("\n线路208对比结果:")
    print(df_208.to_string(index=False))
    
    print("\n线路211对比结果:")
    print(df_211.to_string(index=False))
    
    # 保存CSV
    table_dir = 'results/tables'
    os.makedirs(table_dir, exist_ok=True)
    
    df_208.to_csv(f'{table_dir}/comparison_route_208.csv', 
                  index=False, encoding='utf-8-sig')
    df_211.to_csv(f'{table_dir}/comparison_route_211.csv',
                  index=False, encoding='utf-8-sig')
    
    # 合并表格
    df_208['线路'] = 208
    df_211['线路'] = 211
    df_all = pd.concat([df_208, df_211], ignore_index=True)
    df_all = df_all[['线路', '方法', '方向', '发车次数', 
                     '乘客平均等待时间(分钟)', '被滞留乘客数量', 'ω']]
    
    df_all.to_csv(f'{table_dir}/comparison_all_routes.csv',
                  index=False, encoding='utf-8-sig')
    
    print(f"\n对比表格已保存到: {table_dir}/")
    
    return df_all


def generate_omega_sensitivity_table():
    """
    生成表2-4：DRL-TSBC在不同ω下的测试结果
    """
    print("\n生成ω参数敏感性分析表格...")
    
    # 论文中的数据
    results = [
        {
            'ω': '1/500',
            '线路208_发车次数': 89,
            '线路208_上行AWT': 2.6,
            '线路208_下行AWT': 2.5,
            '线路211_发车次数': 77,
            '线路211_上行AWT': 3.4,
            '线路211_下行AWT': 3.1
        },
        {
            'ω': '1/1000',
            '线路208_发车次数': 77,
            '线路208_上行AWT': 3.5,
            '线路208_下行AWT': 3.3,
            '线路211_发车次数': 68,
            '线路211_上行AWT': 4.8,
            '线路211_下行AWT': 3.9
        },
        {
            'ω': '1/2000',
            '线路208_发车次数': 72,
            '线路208_上行AWT': 3.8,
            '线路208_下行AWT': 4.0,
            '线路211_发车次数': 58,
            '线路211_上行AWT': 6.7,
            '线路211_下行AWT': 5.8
        },
        {
            'ω': '1/3000',
            '线路208_发车次数': 70,
            '线路208_上行AWT': 4.7,
            '线路208_下行AWT': 4.2,
            '线路211_发车次数': 54,
            '线路211_上行AWT': 7.9,
            '线路211_下行AWT': 6.5
        },
        {
            'ω': '1/4000',
            '线路208_发车次数': 64,
            '线路208_上行AWT': 4.9,
            '线路208_下行AWT': 4.9,
            '线路211_发车次数': 52,
            '线路211_上行AWT': 7.8,
            '线路211_下行AWT': 6.5
        }
    ]
    
    df = pd.DataFrame(results)
    
    print("\nω参数敏感性分析结果:")
    print(df.to_string(index=False))
    
    # 保存
    table_dir = 'results/tables'
    os.makedirs(table_dir, exist_ok=True)
    df.to_csv(f'{table_dir}/omega_sensitivity.csv',
              index=False, encoding='utf-8-sig')
    
    print(f"\nω敏感性分析表格已保存到: {table_dir}/omega_sensitivity.csv")
    
    return df


def main():
    """主函数"""
    print("="*70)
    print(" "*20 + "生成对比表格")
    print("="*70)
    
    # 生成性能对比表格
    df_comparison = generate_comparison_table()
    
    # 生成ω敏感性分析表格
    df_omega = generate_omega_sensitivity_table()
    
    print("\n" + "="*70)
    print("所有对比表格生成完成！")
    print("="*70)


if __name__ == '__main__':
    main()
