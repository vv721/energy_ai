import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.config import PAGE_CONFIG, PAGES
from frontend.components import create_navi_menu
from frontend.utils import handle_exc


def main():

    st.set_page_config(**PAGE_CONFIG)

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = 0

    #分隔布局
    left_col, right_col = st.columns([1, 6], gap="small")

    with left_col:
        create_navi_menu(PAGES)

    with right_col:
        selected_page = st.session_state.selected_page

        #查找选中的页面配置
        page_config = next((p for p in PAGES if p["id"] == selected_page), None)

        if page_config:
            if page_config.get("coming_soon", False):
                st.markdown(f"### {page_config['name']}")
                st.info("此功能正在开发中...")
            else:
                try:
                    #动态导入运行页面模块
                    module_name = page_config["module"]
                    module = __import__(f"frontend.{module_name}", fromlist=[module_name])
                    module.main()
                except Exception as e:
                    handle_exc(e, f"运行页面模块 {module_name} 时出错")

if __name__ == "__main__":
    main()
