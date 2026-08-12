// torchtitan learning docs - shared sidebar navigation
// Pure static JS so the docs work offline via file:// (no frameworks).

const PAGES = [
  // 入口
  { file: "index.html",            title: "总览 · 入口" },

  // 架构层
  { file: "01_architecture.html",  title: "整体架构" },
  { file: "architecture.html",     title: "架构(旧版)" },
  { file: "architecture_slide.html",title: "架构幻灯片" },
  { file: "code_structure.html",   title: "代码结构" },
  { file: "torchtitan_tree.html",  title: "目录结构速览" },
  { file: "02_training_flow.html", title: "训练流程(原版)" },
  { file: "07_code_walkthrough.html", title: "代码走读" },

  // 并行技术
  { file: "04_parallelism.html",   title: "并行技术(原版)" },
  { file: "parallelism_code_map.html",title: "代码地图" },
  { file: "parallelism_principles.html",title: "实现原理" },
  { file: "parallelism_sharding_detail.html",title: "切分细节" },
  { file: "parallelism_training_flow.html",title: "并行训练流程" },
  { file: "spmd_backends.html",    title: "SPMD 后端对比" },
  { file: "parallelism_layers.html", title: "并行实现三层" },
  { file: "mesh_example.html",     title: "网格示例" },

  // 训练流程
  { file: "gpu_view_training_flow.html",title: "GPU 视图训练流程" },
  { file: "training_flow_overview.html",title: "训练流程全景" },
  { file: "training_flow_end_to_end.html",title: "端到端训练流程" },

  // 配置与组件
  { file: "03_config_system.html", title: "配置系统" },
  { file: "05_model_system.html",  title: "模型系统" },
  { file: "deepseek_v3_671b_structure.html", title: "DeepSeek V3 671B 结构" },
  { file: "deepseek_v3_671b_training_example.html", title: "DeepSeek V3 671B 训练示例" },
  { file: "06_components.html",    title: "训练组件" },
];

function currentFile() {
  const p = window.location.pathname.split("/").pop();
  return p === "" ? "index.html" : p;
}

function buildSidebar() {
  const cur = currentFile();
  const sb = document.getElementById("sidebar");
  if (!sb) return;

  let html = `<div class="brand">torchtitan 学习文档<small>预训练框架 · 从架构到代码</small></div>`;
  html += `<div class="nav-section">入口</div>`;
  for (const pg of PAGES) {
    const active = pg.file === cur ? " active" : "";
    // 每 6-7 个页面插入一个分类分隔(跳过 index 入口)
    if (pg === PAGES[0]) { /* 入口,已在上方 */ }
    else if (pg === PAGES[1]) {
      html += `<div class="nav-section">架构层</div>`;
    } else if (pg === PAGES[7]) {
      html += `<div class="nav-section">并行技术</div>`;
    } else if (pg === PAGES[15]) {
      html += `<div class="nav-section">训练流程</div>`;
    } else if (pg === PAGES[18]) {
      html += `<div class="nav-section">配置与组件</div>`;
    }
    html += `<a class="nav-link${active}" href="${pg.file}">${pg.title}</a>`;
  }
  html += `<div class="nav-section">快速要点</div>`;
  html += `<a class="nav-link nav-sub" href="01_architecture.html#core-abstractions">核心抽象</a>`;
  html += `<a class="nav-link nav-sub" href="02_training_flow.html#init-ordering">Trainer 初始化顺序</a>`;
  html += `<a class="nav-link nav-sub" href="04_parallelism.html#mesh">设备网格 Mesh</a>`;
  html += `<a class="nav-link nav-sub" href="05_model_system.html#sharding">声明式切分</a>`;
  html += `<a class="nav-link nav-sub" href="07_code_walkthrough.html#init">__init__ 走读</a>`;
  sb.innerHTML = html;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", buildSidebar);
} else {
  buildSidebar();
}
