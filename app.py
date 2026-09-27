import streamlit as st
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="GA Flow-Shop 智能调度系统",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 GA Flow-Shop 智能调度系统")

st.markdown("""
### 置换型流水车间调度问题（Permutation Flow-Shop Scheduling）

已知生产系统包含：

- **6 个工位（设备）：M1 ～ M6**
- **9 个产品**
- **5 种产品类型：A、B、C、D、E**
- 产品组成：**2A + 3B + 1C + 2D + 1E**

本程序使用 **遗传算法（Genetic Algorithm, GA）**
搜索 Makespan 尽可能小的产品加工顺序。
""")

st.divider()


# ============================================================
# 2. Q1：排列组合数量
# ============================================================

st.header("❓ Q1：共有多少种产品排列组合？")

# 产品数量：
# A = 2
# B = 3
# C = 1
# D = 2
# E = 1

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

st.write("""
9 个产品中：

- A 产品有 2 个
- B 产品有 3 个
- C 产品有 1 个
- D 产品有 2 个
- E 产品有 1 个

由于同类型产品之间不区分，因此属于**有重复元素的全排列问题**。
""")

st.write("排列组合数量计算公式：")

st.latex(
    r"N=\frac{9!}{2!\times3!\times1!\times2!\times1!}"
)

st.latex(
    r"N=15120"
)

st.success(
    f"✅ Q1 答案：共有 {num_sequences:,} 种不同的产品排列组合。"
)

st.divider()


# ============================================================
# 3. 加工时间设置
# ============================================================

st.header("⚙️ 产品加工时间设置")

st.markdown("""
下面是 5 种产品分别经过 6 个工位时所需要的加工时间。

**表格中的数字可以直接修改。**

如果加工时间发生变化，只需要修改对应数字，
然后重新点击“开始 GA 智能求解”，程序就会根据新的加工时间重新优化。
""")

machines = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6"
]

# ============================================================
# 白板中的原始加工时间
# ============================================================

default_data = pd.DataFrame(
    {
        "M1": [15, 17, 25, 31, 10],
        "M2": [9, 14, 8, 11, 14],
        "M3": [13, 51, 42, 4, 7],
        "M4": [18, 12, 7, 24, 8],
        "M5": [9, 21, 14, 20, 15],
        "M6": [15, 10, 50, 20, 25]
    },
    index=[
        "A",
        "B",
        "C",
        "D",
        "E"
    ]
)

# ============================================================
# 可编辑表格
# ============================================================

edited_data = st.data_editor(
    default_data,
    use_container_width=True,
    num_rows="fixed",
    key="processing_time_editor"
)

st.info(
    "💡 使用方法：点击表格中的数字即可修改任意产品在任意工位上的加工时间。"
)

# ============================================================
# 检查加工时间
# ============================================================

try:

    if edited_data.isnull().values.any():

        st.error(
            "加工时间表中存在空值，请填写完整。"
        )

        st.stop()

    if (edited_data < 0).values.any():

        st.error(
            "加工时间不能小于 0，请重新输入。"
        )

        st.stop()

except TypeError:

    st.error(
        "加工时间必须输入数字。"
    )

    st.stop()


# ============================================================
# 将表格转换为算法需要的数据
# ============================================================

processing_times = {
    product: [
        float(
            edited_data.loc[
                product,
                machine
            ]
        )
        for machine in machines
    ]
    for product in edited_data.index
}


# ============================================================
# 4. 定义 9 个具体产品
# ============================================================

jobs = [
    "A1",
    "A2",

    "B1",
    "B2",
    "B3",

    "C1",

    "D1",
    "D2",

    "E1"
]


# ============================================================
# 5. Flow-Shop 调度计算函数
# ============================================================

def calculate_schedule(sequence):

    """
    根据给定产品排列计算：
    1. 每个产品在每台设备上的开始时间
    2. 每个产品在每台设备上的完成时间
    3. Makespan
    """

    n_jobs = len(sequence)
    n_machines = len(machines)

    start = np.zeros(
        (
            n_jobs,
            n_machines
        )
    )

    completion = np.zeros(
        (
            n_jobs,
            n_machines
        )
    )

    for i, job in enumerate(sequence):

        # A1 -> A
        # B2 -> B
        product_type = job[0]

        times = processing_times[
            product_type
        ]

        for j in range(n_machines):

            # -----------------------------------------------
            # 第一个产品 + 第一台机器
            # -----------------------------------------------

            if i == 0 and j == 0:

                start[i, j] = 0

            # -----------------------------------------------
            # 第一个产品
            # 必须等它在上一台机器加工完成
            # -----------------------------------------------

            elif i == 0:

                start[i, j] = (
                    completion[
                        i,
                        j - 1
                    ]
                )

            # -----------------------------------------------
            # 第一台机器
            # 必须等前一个产品加工完成
            # -----------------------------------------------

            elif j == 0:

                start[i, j] = (
                    completion[
                        i - 1,
                        j
                    ]
                )

            # -----------------------------------------------
            # 普通情况
            #
            # 当前工序开始时间 =
            #
            # max(
            #     前一个产品离开当前机器的时间,
            #     当前产品离开上一台机器的时间
            # )
            # -----------------------------------------------

            else:

                start[i, j] = max(
                    completion[
                        i - 1,
                        j
                    ],
                    completion[
                        i,
                        j - 1
                    ]
                )

            # -----------------------------------------------
            # 完成时间 = 开始时间 + 加工时间
            # -----------------------------------------------

            completion[i, j] = (
                start[i, j]
                + times[j]
            )

    # 最后一个产品
    # 在最后一台机器上的完成时间
    # 就是 Makespan

    makespan = completion[
        -1,
        -1
    ]

    return (
        makespan,
        start,
        completion
    )


# ============================================================
# 6. 创建初始种群
# ============================================================

def create_population(
    population_size
):

    population = []

    for _ in range(
        population_size
    ):

        chromosome = jobs.copy()

        random.shuffle(
            chromosome
        )

        population.append(
            chromosome
        )

    return population


# ============================================================
# 7. 锦标赛选择
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
        key=lambda chromosome:
        calculate_schedule(
            chromosome
        )[0]
    )

    return candidates[
        0
    ].copy()


# ============================================================
# 8. OX 顺序交叉
# ============================================================

def crossover(
    parent1,
    parent2
):

    size = len(
        parent1
    )

    point1, point2 = sorted(
        random.sample(
            range(size),
            2
        )
    )

    child = [
        None
    ] * size

    # -----------------------------------------------
    # 从父代1保留一段
    # -----------------------------------------------

    child[
        point1:point2
    ] = parent1[
        point1:point2
    ]

    # -----------------------------------------------
    # 按照父代2的顺序
    # 填入剩余基因
    # -----------------------------------------------

    remaining = [
        gene
        for gene in parent2
        if gene not in child
    ]

    index = 0

    for i in range(size):

        if child[i] is None:

            child[i] = (
                remaining[index]
            )

            index += 1

    return child


# ============================================================
# 9. 交换变异
# ============================================================

def mutation(
    chromosome,
    mutation_rate
):

    child = (
        chromosome.copy()
    )

    if (
        random.random()
        < mutation_rate
    ):

        i, j = random.sample(
            range(
                len(child)
            ),
            2
        )

        child[i], child[j] = (
            child[j],
            child[i]
        )

    return child


# ============================================================
# 10. 遗传算法
# ============================================================

def genetic_algorithm(
    population_size,
    generations,
    mutation_rate
):

    # -----------------------------------------------
    # 初始化种群
    # -----------------------------------------------

    population = (
        create_population(
            population_size
        )
    )

    best_solution = None

    best_makespan = (
        float("inf")
    )

    history = []

    # -----------------------------------------------
    # 开始迭代
    # -----------------------------------------------

    for generation in range(
        generations
    ):

        # 按 Makespan 从小到大排序

        population.sort(
            key=lambda chromosome:
            calculate_schedule(
                chromosome
            )[0]
        )

        current_best = (
            population[0]
        )

        current_makespan = (
            calculate_schedule(
                current_best
            )[0]
        )

        # -------------------------------------------
        # 更新全局最优解
        # -------------------------------------------

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

        # -------------------------------------------
        # 精英保留
        #
        # 最好的两个个体
        # 直接进入下一代
        # -------------------------------------------

        new_population = [
            population[0].copy(),
            population[1].copy()
        ]

        # -------------------------------------------
        # 产生剩余个体
        # -------------------------------------------

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

        population = (
            new_population
        )

    return (
        best_solution,
        best_makespan,
        history
    )


# ============================================================
# 11. Q2
# ============================================================

st.divider()

st.header(
    "🧬 Q2：使用遗传算法寻找 Makespan 较小的排列"
)

st.markdown("""
遗传算法将从大量不同的产品排列中进行搜索。

你可以调整下面三个 GA 参数，然后运行算法。
""")


# ============================================================
# 12. GA 参数
# ============================================================

col1, col2, col3 = st.columns(
    3
)

with col1:

    population_size = st.slider(
        "种群规模（Population Size）",
        min_value=20,
        max_value=500,
        value=150,
        step=10
    )

with col2:

    generations = st.slider(
        "迭代次数（Generations）",
        min_value=50,
        max_value=2000,
        value=500,
        step=50
    )

with col3:

    mutation_rate = st.slider(
        "变异概率（Mutation Rate）",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01
    )


# ============================================================
# 13. 开始运行 GA
# ============================================================

if st.button(
    "🚀 开始 GA 智能求解",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "遗传算法正在搜索较优调度方案..."
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
        "✅ 遗传算法搜索完成！"
    )


    # ========================================================
    # 14. Q2 最终结果
    # ========================================================

    st.subheader(
        "🏆 Q2 求解结果"
    )

    result_col1, result_col2 = (
        st.columns(2)
    )

    with result_col1:

        st.metric(
            "当前搜索到的最小 Makespan",
            f"{best_makespan:.0f}"
        )

    with result_col2:

        st.metric(
            "不同类型排列组合数量",
            f"{num_sequences:,}"
        )


    st.write(
        "### GA 搜索到的较优产品排列"
    )

    st.code(
        " → ".join(
            best_sequence
        )
    )

    st.success(
        "Q2 答案："
        + " → ".join(
            best_sequence
        )
        + f"；Makespan = {best_makespan:.0f}"
    )


    # ========================================================
    # 15. 获取完整调度信息
    # ========================================================

    (
        makespan,
        start,
        completion
    ) = calculate_schedule(
        best_sequence
    )


    # ========================================================
    # 16. GA 收敛曲线
    # ========================================================

    st.subheader(
        "📉 GA 收敛曲线"
    )

    st.write(
        "随着迭代次数增加，GA不断寻找 Makespan 更小的排列。"
    )

    fig1, ax1 = plt.subplots(
        figsize=(
            10,
            4
        )
    )

    ax1.plot(
        range(
            1,
            len(history) + 1
        ),
        history
    )

    ax1.set_xlabel(
        "Generation"
    )

    ax1.set_ylabel(
        "Best Makespan"
    )

    ax1.set_title(
        "Genetic Algorithm Convergence"
    )

    ax1.grid(
        alpha=0.3
    )

    st.pyplot(
        fig1
    )

    plt.close(
        fig1
    )


    # ========================================================
    # 17. Flow-Shop 甘特图
    # ========================================================

    st.subheader(
        "📊 Flow-Shop 调度甘特图"
    )

    st.write("""
甘特图表示每一个产品在每一台机器上的加工时间段。

横轴表示时间，纵轴表示机器。
""")

    fig2, ax2 = plt.subplots(
        figsize=(
            14,
            7
        )
    )

    for i, job in enumerate(
        best_sequence
    ):

        for j, machine in enumerate(
            machines
        ):

            start_time = (
                start[i, j]
            )

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
        range(
            len(machines)
        )
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
        "Flow-Shop Schedule"
        + f" | Makespan = {makespan:.0f}"
    )

    ax2.grid(
        axis="x",
        alpha=0.2
    )

    ax2.invert_yaxis()

    st.pyplot(
        fig2
    )

    plt.close(
        fig2
    )


    # ========================================================
    # 18. 开始时间矩阵
    # ========================================================

    st.subheader(
        "⏱️ 各工序开始时间"
    )

    start_df = pd.DataFrame(
        start,
        index=best_sequence,
        columns=machines
    )

    st.dataframe(
        start_df.round(0),
        use_container_width=True
    )


    # ========================================================
    # 19. 完工时间矩阵
    # ========================================================

    st.subheader(
        "📋 各工序完工时间"
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
# 20. 算法说明
# ============================================================

st.divider()

st.header(
    "📖 算法说明"
)

st.markdown("""
### 1. 什么是 Flow-Shop 调度？

在本问题中，每一个产品都需要依次经过：

**M1 → M2 → M3 → M4 → M5 → M6**

六个工位。

同一时间，一台机器只能加工一个产品。

同时，一个产品必须完成前一道工序后，
才能进入下一台机器继续加工。
""")


# ============================================================
# 染色体
# ============================================================

st.markdown("""
### 2. 什么是染色体？

遗传算法需要把一个调度方案表示成一个“染色体”。

例如：
""")

st.code(
    "A1 → B1 → D1 → A2 → C1 → B2 → E1 → D2 → B3"
)

st.markdown("""
上面的染色体就代表：

**这 9 个产品按照这个顺序依次进入生产线。**

不同的染色体代表不同的产品排列方案。
""")


# ============================================================
# GA过程
# ============================================================

st.markdown("""
### 3. 遗传算法如何寻找更好的排列？

本程序主要执行以下步骤：

**① 初始化种群**

随机产生大量不同的产品排列。

↓

**② 锦标赛选择**

优先选择 Makespan 较小的排列作为父代。

↓

**③ OX 顺序交叉**

将两个父代的排列信息进行组合，产生新的子代。

↓

**④ 交换变异**

随机交换两个产品的位置，提高种群多样性。

↓

**⑤ 精英保留**

当前最好的调度方案直接进入下一代，避免优秀方案丢失。

↓

**⑥ 重复迭代**

不断重复选择、交叉和变异，
逐渐寻找 Makespan 更小的产品排列。
""")


# ============================================================
# Makespan解释
# ============================================================

st.markdown("""
### 4. 什么是 Makespan？

Makespan 可以理解为：

> **从第一个产品开始加工，到最后一个产品完成最后一道工序所需要的总时间。**

也就是最后一个产品在最后一台机器 M6 上的完成时间。
""")


# ============================================================
# 正确渲染公式
# ============================================================

st.write(
    "本问题的优化目标是："
)

st.latex(
    r"\min \; C_{\mathrm{max}}"
)

st.write(
    "其中："
)

st.latex(
    r"C_{\mathrm{max}}=\text{最后一个产品在最后一个工位上的完成时间}"
)

st.markdown("""
也就是说：

### **我们的目标就是让 Makespan 尽可能小。**
""")


# ============================================================
# 简单例子
# ============================================================

st.markdown("""
### 5. 为什么 Makespan 越小越好？

例如有两个调度方案：

- 方案 A：Makespan = **350**
- 方案 B：Makespan = **310**

因为：
""")

st.latex(
    r"310 < 350"
)

st.success(
    "因此方案 B 能够更早完成全部产品的加工，调度效果更好。"
)


# ============================================================
# 最终总结
# ============================================================

st.markdown("""
### 6. 本程序的求解逻辑

简单来说，本程序所做的事情就是：

**随机产生很多产品排列**

↓

**计算每一种排列的 Makespan**

↓

**保留较好的排列**

↓

**通过选择、交叉和变异继续产生新的排列**

↓

**经过多代搜索**

↓

### 🏆 找到 Makespan 尽可能小的产品排列
""")
