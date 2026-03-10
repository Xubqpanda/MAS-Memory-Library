MAS-Memory-Library:
├── assets/                                      # 相关资源，如图示等
├── docs/                                        # 文档和设计说明
├── experiments/                                 # 实验脚本和配置
│   ├── configs/                                 # 配置文件
|   |   ├── benchmarks                           # 基准配置文件  
|   |   |    └── frontier_science.yaml           # FrontierScience 基准的实验配置
|   |   └── method                               # 方法配置文件 
|   |        └── single_agent_emptymemory.yaml   # 单智能体的空记忆方法的实验配置   
│   ├── logs/                                    # 运行日志
│   ├── results/                                 # 实验结果和分析
│   ├── scripts/                                 # 跑各个基准数据集的脚本
|   |   └── run_FrontierScience.sh               # 运行 FrontierScience 基准的脚本
│   ├── benchmarks /                             # 评测基准数据集, 包含数据加载和预处理的代码
|   │   |   ├── ALFWorld/                        # ALFWorld 基准   
|   │   |   ├── Fever/                           # Fever 基准   
|   │   |   ├── FrontierScience/                 # FrontierScience 基准   
|   |   |   |   ├── data/                        # FrontierScience 基准的数据文件
|   |   |   |   └── runner.py                    # FrontierScience 基准的运行脚本，包含数据加载和预处理的代码 
|   │   |   ├── HLE/                             # HLE 基准 
|   |   |   |   ├── data/                        # HLE 基准的数据文件
|   |   |   |   └── runner.py                    # HLE 基准的运行脚本，包含数据加载和预处理的代码 
|   │   |   └── PDDL/                            # PDDL 基准
│   └── run_experiment.py                        # 统一的实验运行入口，支持不同配置的实验  
├── src/                                         # 核心代码库
│   ├── common/                                  # 跨层共享的数据结构
│   │   ├── __init__.py                          # common 模块的初始化
│   │   └── message.py                           # MASMessage, AgentMessage, TaskResult
│   ├── envs/                                    # 任务处理的环境
│   │   ├── __init__.py                          # envs 模块的初始化
│   │   ├── alfworld.py                          # alfworld 任务环境配置
│   │   ├── base.py                              # envs 模块基类
│   │   ├── fever.py                             # fever 任务环境配置
│   │   ├── frontierscience.py                   # frontierscience 任务环境配置
│   │   ├── hle.py                               # hle 任务环境配置
│   │   └── pddl.py                              # pddl 任务环境配置
│   ├── llm/                                     # LLM 相关的接口和实现
│   │   ├── __init__.py                          # llm 模块的初始化
│   │   ├── base.py                              # LLMBase 抽象类
│   │   ├── llm_io_logger.py                     # LLM I/O 日志记录器
│   │   ├── model_caller.py                      # ModelCaller 实现
│   │   └── token_tracker.py                     # TokenTracker 实现
│   ├── solver/                                  # 智能体系统相关的接口和实现
│   │   ├── autogen/                             # AutoGenMAS(MetaSolver)
│   │   │   ├── __init__.py                      # AutoGenMAS 模块的初始化
│   │   │   ├── autogen_prompt.py                # AutoGenMAS(MetaSolver) 的提示词设计
│   │   │   └── autogen.py                       # AutoGenMAS(MetaSolver) 的核心实现
│   │   ├── dylan/                               # DylanMAS(MetaSolver)
│   │   │   ├── __init__.py                      # DylanMAS 模块的初始化
│   │   │   ├── dylan_prompt.py                  # DylanMAS(MetaSolver) 的提示词设计
│   │   │   ├── dylan.py                         # DylanMAS(MetaSolver) 的核心实现  
│   │   │   └── neuron.py                        # DylanMAS(MetaSolver) 的神经元实现
│   │   ├── macnet/                              # MacNetMAS(MetaSolver)
│   │   │   ├── __init__.py                      # MacNetMAS 模块的初始化
│   │   │   ├── graph_mas.py                     # MacNetMAS(MetaSolver) 的核心实现
│   │   │   ├── graph_prompt.py                  # MacNetMAS(MetaSolver) 的提示词设计
│   │   │   ├── graph.py                         # MacNetMAS(MetaSolver) 的图结构实现
│   │   │   └── node.py                          # MacNetMAS(MetaSolver) 的节点实现
|   │   ├── single_agent/                        # SingleAgentSolver(MetaSolver)
│   │   │   ├── __init__.py                      # SingleAgentSolver 模块的初始化
│   │   │   └── single_agent.py                  # SingleAgentSolver(MetaSolver) 的核心实现
│   │   ├── __init__.py                          # MAS 相关的抽象和接口设计
│   │   ├── format.py                            # 任务输入输出的格式规范
│   │   └── base.py                              # MetaSolver 抽象类
│   ├─ reasoning/                                # Reasoning
|   |   ├── __init__.py                          # Reasoning 模块的初始化
|   |   └── base.py                              # ReasoningConfig、ReasoningBase、ReasoningIO
│   ├── memory/                                  # Memory 相关的接口和实现
│   │   ├── methods/                             # 各类 Memory 方法的实现
│   │   |   ├── __init__.py                      # methods 模块的初始化
│   │   |   ├── chatdev.py                       # ChatDevMASMemory 实现
│   │   |   ├── empty.py                         # EmptyMemory 实现
│   │   |   ├── generative.py                    # Generative Memory 实现
│   │   |   ├── GMemory.py                       # GMemory 实现
│   │   |   ├── memory_base.py                   # MemoryBase 抽象类
│   │   |   ├── memorybank.py                    # MemoryBankMASMemory 实现
│   │   |   ├── metagpt.py                       # MetaGPTMASMemory 实现
│   │   |   ├── voyager.py                       # VoyagerMASMemory 实现
│   │   |   └── skillmem/                        # ours 实现   
│   │   ├── __init__.py                          # memory 模块的初始化
│   │   ├── base.py                              # MemoryBase 抽象类
│   │   └── prompt.py                            # Memory 相关的提示词设计
│   ├── registry/                                # 各类组件的注册表设计 
|   |   ├── __init__.py                          # registry 模块的初始化
|   |   └── registry.py                          # MAS/Memory/Reasoning 注册表
|   └── utils/                                   # 各类实用工具函数
|       ├── __init__.py                          # utils 模块的初始化
|       └── helpers.py                           # 常用的辅助函数
├── .gitignore                                   # git 忽略文件
├── LICENSE                                      # 许可证
├── README.md                                    # 项目简介和使用说明
└── requirements.txt                             # 环境依赖