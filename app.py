import streamlit as st
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="GA Flow-Shop 智能调度系统",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 GA Flow-Shop 智能调度系统")

st.markdown("""
### 置换型流水车间调度问题

假设：

- 6 个工位：M1～M6
- 9 个产品
- 5 种产品类型
- 产品组成：**2A + 3B + 1C + 2D + 1E**

使用 **遗传算法（Genetic Algorithm, GA）**
搜索 Makespan 尽可能小的产品排列。
""")

# ============================================================
# Q1：排列组合
# ============================================================

st.header("❓ Q1：一共有多少种排列组合？")

# 9! / (2! 3! 1! 2! 1!)
num_sequences = (
    math.factorial(9)
    //
    (
        math.factorial(2)
        * math.factorial(3)
        * math.factorial(1)
        * math.factorial(2)
        * math.factorial(1)
    )
)

col1, col2 = st.columns([2, 1])

with col1:

    st.latex(
        r"N=\frac{9!}{2!\times3!\times1!\times2!\times1!}"
    )

with col2:

    st.metric(
        "不同排列组合数量",
        f"{num_sequences:,} 种"
    )

st.success(
    f"Q1答案：共有 {num_sequences:,} 种不同的产品排列组合。"
)

st.divider()

# ============================================================
# 加工时间
# ============================================================

st.header("⚙️ 产品加工时间设置")

st.write(
    "下面的加工时间可以直接修改。修改后，GA会按照新的加工时间重新求解。"
)

machines = [
    "M1", "M2", "M3",
    "M4", "M5", "M6"
]

# 白板原始数据
default_data = pd.DataFrame(
    {
        "M1": [15, 17, 25, 31, 10],
        "M2": [9, 14, 8, 11, 14],
        "M3": [13, 51, 42, 4, 7],
        "M4": [18, 12, 7, 24, 8],
        "M5": [9, 21, 14, 20, 15],
        "M6": [15, 10, 50, 20, 25]
    },
    index=["A", "B", "C", "D", "E"]
)

# 可编辑表格
edited_data = st.data_editor(
    default_data,
    use_container_width=True,
    num_rows="fixed",
    key="processing_table"
)

# 转换为程序使用的数据
processing_times = {
    product: [
        float(edited_data.loc[product, machine])
        for machine in machines
    ]
    for product in edited_data.index
}

st.info(
    "💡 你可以双击表格中的数字，修改任意产品在任意工位上的加工时间。"
)

# ============================================================
# 产品
# ============================================================

jobs = [
    "A1", "A2",
    "B1", "B2", "B3",
    "C1",
    "D1", "D2",
    "E1"
]

# ============================================================
# Flow-Shop 调度计算
# ============================================================

def calculate_schedule(sequence):

    n_jobs = len(sequence)
    n_machines = len(machines)

    start = np.zeros(
        (n_jobs, n_machines)
    )

    completion = np.zeros(
        (n_jobs, n_machines)
    )

    for i, job in enumerate(sequence):

        product_type = job[0]

        times = processing_times[
            product_type
        ]

        for j in range(n_machines):

            if i == 0 and j == 0:

                start[i, j] = 0

            elif i == 0:

                start[i, j] = (
                    completion[i, j - 1]
                )

            elif j == 0:

                start[i, j] = (
                    completion[i - 1, j]
                )

            else:

                start[i, j] = max(
                    completion[i - 1, j],
                    completion[i, j - 1]
                )

            completion[i, j] = (
                start[i, j]
                + times[j]
            )

    makespan = completion[-1, -1]

    return (
        makespan,
        start,
        completion
    )

# ============================================================
# 创建初始种群
# ============================================================

def create_population(population_size):

    population = []

    for _ in range(population_size):

        chromosome = jobs.copy()

        random.shuffle(chromosome)

        population.append(chromosome)

    return population

# ============================================================
# 锦标赛选择
# ============================================================

def tournament_selection(
    population,
    tournament_size=3
):

    candidates = random.sample(
        population,
        tournament_size
    )

    candidates.sort(
        key=lambda x:
        calculate_schedule(x)[0]
    )

    return candidates[0].copy()

# ============================================================
# OX顺序交叉
# ============================================================

def crossover(parent1, parent2):

    size = len(parent1)

    point1, point2 = sorted(
        random.sample(
            range(size),
            2
        )
    )

    child = [None] * size

    child[
        point1:point2
    ] = parent1[
        point1:point2
    ]

    remaining = [
        gene
        for gene in parent2
        if gene not in child
    ]

    index = 0

    for i in range(size):

        if child[i] is None:

            child[i] = remaining[index]

            index += 1

    return child

# ============================================================
# 变异
# ============================================================

def mutation(
    chromosome,
    mutation_rate
):

    child = chromosome.copy()

    if random.random() < mutation_rate:

        i, j = random.sample(
            range(len(child)),
            2
        )

        child[i], child[j] = (
            child[j],
            child[i]
        )

    return child

# ============================================================
# GA
# ============================================================

def genetic_algorithm(
    population_size,
    generations,
    mutation_rate
):

    population = create_population(
        population_size
    )

    best_solution = None

    best_makespan = float("inf")

    history = []

    for generation in range(
        generations
    ):

        population.sort(
            key=lambda x:
            calculate_schedule(x)[0]
        )

        current_best = population[0]

        current_makespan = (
            calculate_schedule(
                current_best
            )[0]
        )

        if (
            current_makespan
            < best_makespan
        ):

            best_makespan = (
                current_makespan
            )

            best_solution = (
                current_best.copy()
            )

        history.append(
            best_makespan
        )

        # 精英保留
        new_population = [
            population[0].copy(),
            population[1].copy()
        ]

        while (
            len(new_population)
            < population_size
        ):

            parent1 = (
                tournament_selection(
                    population
                )
            )

            parent2 = (
                tournament_selection(
                    population
                )
            )

            child = crossover(
                parent1,
                parent2
            )

            child = mutation(
                child,
                mutation_rate
            )

            new_population.append(
                child
            )

        population = new_population

    return (
        best_solution,
        best_makespan,
        history
    )

# ============================================================
# Q2
# ============================================================

st.divider()

st.header(
    "🧬 Q2：使用GA求Makespan较小的排列组合"
)

st.write(
    "设置遗传算法参数，然后点击运行。"
)

# ============================================================
# GA 参数
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    population_size = st.slider(
        "种群规模",
        min_value=20,
        max_value=500,
        value=150,
        step=10
    )

with col2:

    generations = st.slider(
        "迭代次数",
        min_value=50,
        max_value=2000,
        value=500,
        step=50
    )

with col3:

    mutation_rate = st.slider(
        "变异概率",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01
    )

# ============================================================
# 运行
# ============================================================

if st.button(
    "🚀 开始GA智能求解",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "GA正在搜索更小的Makespan..."
    ):

        (
            best_sequence,
            best_makespan,
            history
        ) = genetic_algorithm(
            population_size,
            generations,
            mutation_rate
        )

    st.success(
        "GA搜索完成！"
    )

    # ========================================================
    # Q2最终答案
    # ========================================================

    st.subheader(
        "🏆 Q2 求解结果"
    )

    result1, result2 = st.columns(2)

    with result1:

        st.metric(
            "最小 Makespan",
            f"{best_makespan:.0f}"
        )

    with result2:

        st.metric(
            "搜索的排列空间",
            f"{num_sequences:,} 种"
        )

    st.write(
        "### GA找到的较优产品排列"
    )

    st.code(
        " → ".join(best_sequence)
    )

    st.success(
        "Q2答案："
        + " → ".join(best_sequence)
        + f"；Makespan = {best_makespan:.0f}"
    )

    # ========================================================
    # 调度数据
    # ========================================================

    (
        makespan,
        start,
        completion
    ) = calculate_schedule(
        best_sequence
    )

    # ========================================================
    # 收敛曲线
    # ========================================================

    st.subheader(
        "📉 GA 收敛曲线"
    )

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
        "GA Convergence"
    )

    ax1.grid(
        alpha=0.3
    )

    st.pyplot(fig1)

    plt.close(fig1)

    # ========================================================
    # 甘特图
    # ========================================================

    st.subheader(
        "📊 Flow-Shop 甘特图"
    )

    fig2, ax2 = plt.subplots(
        figsize=(14, 7)
    )

    for i, job in enumerate(
        best_sequence
    ):

        for j, machine in enumerate(
            machines
        ):

            start_time = start[i, j]

            duration = (
                completion[i, j]
                - start[i, j]
            )

            ax2.barh(
                j,
                duration,
                left=start_time
            )

            ax2.text(
                start_time
                + duration / 2,
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
        "Flow-Shop Schedule "
        f"| Makespan = {makespan:.0f}"
    )

    ax2.grid(
        axis="x",
        alpha=0.2
    )

    ax2.invert_yaxis()

    st.pyplot(fig2)

    plt.close(fig2)

    # ========================================================
    # 完工时间矩阵
    # ========================================================

    st.subheader(
        "📋 完工时间矩阵"
    )

    completion_df = pd.DataFrame(
        completion,
        index=best_sequence,
        columns=machines
    )

    st.dataframe(
        completion_df.round(0),
        use_container_width=True
    )

# ============================================================
# 算法说明
# ============================================================

st.divider()

st.subheader("📖 算法说明")

st.markdown("""
本程序采用遗传算法求解置换型 Flow-Shop 调度问题。

一个染色体代表一种产品排列，例如：

`A1 → B1 → D1 → A2 → C1 → ...`

遗传算法主要过程：

**初始化种群 → 锦标赛选择 → OX顺序交叉 → 交换变异 → 精英保留 → 产生下一代**

优化目标为：

\[
\min C_{max}
\]

其中 \(C_{max}\) 为最后一个产品在最后一个工位上的完工时间，即 Makespan。

因此，**Makespan 越小，调度方案越好。**
""")
