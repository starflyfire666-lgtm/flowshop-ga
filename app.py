import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="GA Flow-Shop 智能调度系统",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 基于遗传算法的 Flow-Shop 智能调度系统")

st.markdown("""
### 作业二：Permutation Flow-Shop Scheduling Problem

使用 **Genetic Algorithm（GA，遗传算法）**，
寻找使最大完工时间 **Makespan 最小** 的产品排列组合。

本系统包含：

- 遗传算法自动搜索
- 最优产品排列
- Makespan 计算
- GA 收敛曲线
- Flow-Shop 甘特图
""")


# ============================================================
# 1. 原始数据
# ============================================================

# 产品加工时间
# 顺序：M1 M2 M3 M4 M5 M6

processing_times = {

    "A": [15, 9, 13, 18, 9, 15],

    "B": [17, 14, 51, 12, 21, 10],

    "C": [25, 8, 42, 7, 14, 50],

    "D": [31, 11, 4, 24, 20, 20],

    "E": [10, 14, 7, 8, 15, 25]

}

# 作业要求：
# 2A + 3B + 1C + 2D + 1E

jobs = [
    "A1", "A2",
    "B1", "B2", "B3",
    "C1",
    "D1", "D2",
    "E1"
]

machines = ["M1", "M2", "M3", "M4", "M5", "M6"]


# ============================================================
# 2. Makespan计算
# ============================================================

def calculate_schedule(sequence):

    n_jobs = len(sequence)
    n_machines = len(machines)

    completion = np.zeros((n_jobs, n_machines))

    start = np.zeros((n_jobs, n_machines))

    for i, job in enumerate(sequence):

        product_type = job[0]

        times = processing_times[product_type]

        for j in range(n_machines):

            # 第一件产品 + 第一台机器
            if i == 0 and j == 0:

                start[i][j] = 0

            # 第一件产品
            elif i == 0:

                start[i][j] = completion[i][j - 1]

            # 第一台机器
            elif j == 0:

                start[i][j] = completion[i - 1][j]

            # 一般情况
            else:

                start[i][j] = max(
                    completion[i - 1][j],
                    completion[i][j - 1]
                )

            completion[i][j] = (
                start[i][j] + times[j]
            )

    makespan = completion[-1][-1]

    return makespan, start, completion


# ============================================================
# 3. 创建初始种群
# ============================================================

def create_population(population_size):

    population = []

    for _ in range(population_size):

        chromosome = jobs.copy()

        random.shuffle(chromosome)

        population.append(chromosome)

    return population


# ============================================================
# 4. 适应度函数
# ============================================================

def fitness(chromosome):

    makespan, _, _ = calculate_schedule(chromosome)

    # Makespan越小越好
    return 1 / makespan


# ============================================================
# 5. 锦标赛选择
# ============================================================

def tournament_selection(population, tournament_size=3):

    candidates = random.sample(
        population,
        tournament_size
    )

    candidates.sort(
        key=lambda x: calculate_schedule(x)[0]
    )

    return candidates[0].copy()


# ============================================================
# 6. 顺序交叉 OX
# ============================================================

def crossover(parent1, parent2):

    size = len(parent1)

    point1, point2 = sorted(
        random.sample(range(size), 2)
    )

    child = [None] * size

    # 保留父代1的一部分
    child[point1:point2] = parent1[point1:point2]

    # 父代2剩余基因
    remaining = [
        gene for gene in parent2
        if gene not in child
    ]

    index = 0

    for i in range(size):

        if child[i] is None:

            child[i] = remaining[index]

            index += 1

    return child


# ============================================================
# 7. 变异
# ============================================================

def mutation(chromosome, mutation_rate):

    child = chromosome.copy()

    if random.random() < mutation_rate:

        i, j = random.sample(
            range(len(child)),
            2
        )

        child[i], child[j] = child[j], child[i]

    return child


# ============================================================
# 8. 遗传算法
# ============================================================

def genetic_algorithm(
        population_size,
        generations,
        mutation_rate
):

    population = create_population(population_size)

    best_history = []

    best_solution = None

    best_makespan = float("inf")

    for generation in range(generations):

        # 按 Makespan 排序
        population.sort(
            key=lambda x: calculate_schedule(x)[0]
        )

        current_best = population[0]

        current_makespan = calculate_schedule(
            current_best
        )[0]

        # 更新全局最优
        if current_makespan < best_makespan:

            best_makespan = current_makespan

            best_solution = current_best.copy()

        best_history.append(best_makespan)

        # ====================================================
        # 精英保留
        # ====================================================

        new_population = [
            population[0].copy(),
            population[1].copy()
        ]

        # ====================================================
        # 产生下一代
        # ====================================================

        while len(new_population) < population_size:

            parent1 = tournament_selection(population)

            parent2 = tournament_selection(population)

            child = crossover(parent1, parent2)

            child = mutation(
                child,
                mutation_rate
            )

            new_population.append(child)

        population = new_population

    return (
        best_solution,
        best_makespan,
        best_history
    )


# ============================================================
# 9. 左侧控制栏
# ============================================================

st.sidebar.header("⚙️ GA 参数")

population_size = st.sidebar.slider(
    "种群规模 Population Size",
    20,
    500,
    150
)

generations = st.sidebar.slider(
    "迭代次数 Generations",
    50,
    2000,
    500
)

mutation_rate = st.sidebar.slider(
    "变异概率 Mutation Rate",
    0.01,
    0.50,
    0.15
)


# ============================================================
# 10. 显示加工时间
# ============================================================

st.subheader("📊 产品加工时间")

df_time = pd.DataFrame(
    processing_times,
    index=machines
).T

st.dataframe(
    df_time,
    use_container_width=True
)

st.write(
    "**产品数量：** 2A + 3B + 1C + 2D + 1E = 9 个产品"
)


# ============================================================
# 11. 开始运行GA
# ============================================================

if st.button(
    "🚀 运行遗传算法",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "遗传算法正在搜索最优调度方案..."
    ):

        best_sequence, best_makespan, history = (
            genetic_algorithm(
                population_size,
                generations,
                mutation_rate
            )
        )

    # ========================================================
    # 最优结果
    # ========================================================

    st.success("遗传算法运行完成！")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "最优 Makespan",
            f"{int(best_makespan)}"
        )

    with col2:

        st.metric(
            "产品数量",
            len(best_sequence)
        )

    st.subheader("🏆 GA 找到的最优排列")

    st.code(
        " → ".join(best_sequence)
    )


    # ========================================================
    # 12. 收敛曲线
    # ========================================================

    st.subheader("📉 GA 收敛曲线")

    fig1, ax1 = plt.subplots(
        figsize=(10, 4)
    )

    ax1.plot(history)

    ax1.set_xlabel(
        "Generation"
    )

    ax1.set_ylabel(
        "Best Makespan"
    )

    ax1.set_title(
        "Genetic Algorithm Convergence"
    )

    ax1.grid(alpha=0.3)

    st.pyplot(fig1)


    # ========================================================
    # 13. 计算甘特图数据
    # ========================================================

    makespan, start, completion = (
        calculate_schedule(best_sequence)
    )


    # ========================================================
    # 14. 甘特图
    # ========================================================

    st.subheader("📅 Flow-Shop 甘特图")

    fig2, ax2 = plt.subplots(
        figsize=(14, 7)
    )

    for i, job in enumerate(best_sequence):

        for j, machine in enumerate(machines):

            start_time = start[i][j]

            duration = (
                completion[i][j]
                - start[i][j]
            )

            ax2.barh(
                j,
                duration,
                left=start_time
            )

            ax2.text(
                start_time + duration / 2,
                j,
                job,
                ha="center",
                va="center",
                fontsize=8
            )

    ax2.set_yticks(
        range(len(machines))
    )

    ax2.set_yticklabels(
        machines
    )

    ax2.set_xlabel(
        "Time"
    )

    ax2.set_ylabel(
        "Machine"
    )

    ax2.set_title(
        f"Flow-Shop Schedule | Makespan = {int(makespan)}"
    )

    ax2.grid(
        axis="x",
        alpha=0.2
    )

    ax2.invert_yaxis()

    st.pyplot(fig2)


    # ========================================================
    # 15. 完工时间矩阵
    # ========================================================

    st.subheader("📋 完工时间矩阵")

    completion_df = pd.DataFrame(
        completion.astype(int),
        index=best_sequence,
        columns=machines
    )

    st.dataframe(
        completion_df,
        use_container_width=True
    )


# ============================================================
# 页面说明
# ============================================================

st.divider()

st.markdown("""
### 🧠 算法原理

染色体表示一个产品加工排列，例如：

`A1 → B2 → D1 → C1 → ...`

GA 不断进行：

**初始化种群 → 选择 → 交叉 → 变异 → 精英保留 → 下一代**

目标函数：

\[
\min C_{max}
\]

其中 \(C_{max}\) 为最后一个产品在最后一台设备上的完工时间，
即 **Makespan**。

Makespan 越小，说明整个生产系统完成全部产品所需要的时间越短。
""")
