# 用Altair绘制各种数据图表
import altair as alt
from pandas import DataFrame, Series
from typing import Union, Optional
alt.data_transformers.enable("vegafusion")


def check_if_single_column(data: Union[DataFrame, Series]) -> bool:
    """) -> bool:
    检查数据是否是Series或只有一列的DataFrame。

    :param data: 待检查的数据
    :return:  是否只有一列
    """
    if isinstance(data, Series):
        return True
    elif isinstance(data, DataFrame):
        if len(data.columns) == 1:
            return True
        else:
            return False
    else:
        return False


def melt_data(data: Union[DataFrame, Series],
              x_col: Optional[str] = None,
              ) -> DataFrame:
    """
    将数据转换为长格式，以适应altair的绘图要求。

    :param data: 待转换的数据
    :param x_col: x轴列名，如果为None，则使用索引作为x轴，如果data是Series或者单列DataFrame，则忽略此参数
    :return:  转换后的数据，一个三列的dataframe。x_col列为x轴，其它列名将转到"category"列，值将转到"y"列
    """
    if check_if_single_column(data):  # 如果是Series或只有一列的DataFrame
        data: DataFrame = data.reset_index()
        data = data.rename(columns={data.columns[0]: 'x'})
    else:
        if x_col is None:
            data = data.reset_index()
            data = data.rename(columns={data.columns[0]: 'x'})
        elif x_col not in data.columns:
            raise ValueError(f'x_col参数"{x_col}"不在数据的列名中')
        else:
            data = data.rename(columns={x_col: 'x'})
    return data.melt(id_vars='x', var_name='category', value_name='y')


def plot_line(data: Union[DataFrame, Series],
              x_col: Optional[str] = None,
              selector: bool = True,
              ) -> alt.Chart:
    """
    绘制单个或多个特征的折线图，支持交互式图表。

    :param data: 用于绘图的数据
    :param x_col: x轴列名，如果为None，则使用索引作为x轴
    :param selector: 选择器，用于交互式图表
    :return:  图表对象
    """
    # 将数据转换为长格式
    data = melt_data(data, x_col)
    # 检查x列的数据类型是否是时间
    x_is_time = True if data['x'].dtype == 'datetime64[ns]' else False
    # 如果x轴对应的数据是时间格式，则设置x轴的类型为时间
    x = alt.X('x', axis=alt.Axis(title=None,
                                 format='%Y-%m-%d'
                                 ) if x_is_time else alt.Axis(title=None)
              )
    # 设置y轴无标题
    y = alt.Y('y', axis=alt.Axis(title=None))
    # 设置悬停信息
    tooltip = [alt.Tooltip('x', title='x', format='%Y-%m-%d'
                           ) if x_is_time else alt.Tooltip('x', title='x'),
               alt.Tooltip('y', title='y')]
    # 绘制最基础的折线图
    chart = alt.Chart(data).mark_line().encode(x=x, y=y,
                                               color=alt.Color('category').title(''),
                                               tooltip=tooltip
                                               )
    # 透明选择器
    if selector:
        # 创建一个最近的选择器
        nearest = alt.selection_point(nearest=True, on='mouseover',
                                      fields=['x'], empty=False)
        selectors = alt.Chart(data).mark_point().encode(
            x=x,
            y=y,
            opacity=alt.value(0)
        ).add_params(nearest)  # 设置光标的x值
        # 绘制选择器选中的点
        points = chart.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
        # 显示文本
        text = chart.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'y:Q', alt.value(' '))
        )
        # 绘制选择器的位置
        rules = alt.Chart(data).mark_rule(color='gray').encode(
            x=x
        ).transform_filter(nearest)
        # 组合图表
        chart = alt.layer(chart, selectors, points, rules, text)
    chart = chart.properties(
        width=1000,
        height=500
    ).interactive()
    return chart


def plot_bar(data: Union[DataFrame, Series]) -> alt.Chart:
    """
    绘制单个特征的柱状图，支持交互式图表。

    :param data: 用于绘图的数据，index为x轴，values为y轴，只能设置为Series或单列DataFrame
    :return:  图表对象
    """
    # 数据格式转换
    if not check_if_single_column(data):
        raise ValueError('数据必须为Series或单列DataFrame')
    data = melt_data(data)
    # 绘制柱状图
    x = alt.X('x',
              axis=alt.Axis(title=None,
                            format='%Y-%m-%d %H:%M',
                            ) if data['x'].dtype == 'datetime64[ns]' else alt.Axis(title=None),
              scale=alt.Scale(padding=0)
              )
    y = alt.Y('y', axis=alt.Axis(title=None))
    tooltip = [alt.Tooltip('x', title='x', format='%Y-%m-%d %H:%M'
                           ) if data['x'].dtype == 'datetime64[ns]' else alt.Tooltip('x', title='x'),
               alt.Tooltip('y', title='y')
               ]
    chart = alt.Chart(data).mark_bar(size=8, binSpacing=0).encode(
        x=x,
        y=y,
        tooltip=tooltip
    )
    chart = chart.properties(
        width=1000,
        height=500
    ).interactive()
    return chart


def plot_runoff_and_rainfall(runoff: Union[DataFrame, Series],
                             rainfall: Union[DataFrame, Series],
                             title: Optional[str] = None,
                             ) -> alt.Chart:
    """
    绘制径流和降雨的折线图。

    :param runoff: 径流数据
    :param rainfall: 降雨数据
    :param title: 图表标题，推荐写NSE等评价指标
    :return:  图表对象
    """
    # 检查降水和径流数据的index是否是时间
    if not runoff.index.dtype == 'datetime64[ns]':
        raise ValueError('径流数据的index必须是时间')
    if not rainfall.index.dtype == 'datetime64[ns]':
        raise ValueError('降雨数据的index必须是时间')
    runoff_chart = plot_line(runoff, selector=False)
    rainfall_chart = plot_bar(rainfall)
    # 将runoff_chart和rainfall_chart的y轴都设置为[0, 原来的最大值的2倍]
    if isinstance(runoff, Series):
        runoff_max = runoff.max()
    else:
        runoff_max = runoff.max().max()
    rainfall_max = rainfall.max()
    runoff_chart = runoff_chart.encode(
        y=alt.Y('y',
                axis=alt.Axis(title=None),
                scale=alt.Scale(domain=[0, runoff_max * 2]),
                )
    )
    rainfall_chart = rainfall_chart.encode(y=alt.Y('y',
                                                   axis=alt.Axis(title=None, orient='right', ),
                                                   scale=alt.Scale(reverse=True, domain=[0, rainfall_max * 2]),
                                                   ))
    # 组合图表
    chart = alt.layer(runoff_chart, rainfall_chart).resolve_scale(y='independent').interactive(bind_y=False)
    # 设置图表标题
    if title is not None:
        chart = chart.properties(title=title)
    return chart


def plot_single_histogram(data: Union[DataFrame, Series]) -> alt.Chart:
    """
    绘制单个特征的直方图。

    :param data: 用于绘图的数据，只能设置为Series或单列DataFrame
    :return: 图表对象
    """
    if not check_if_single_column(data):
        raise ValueError('数据必须为Series或单列DataFrame')
    if isinstance(data, Series):  # 如果是Series，则转换为DataFrame
        if data.name is None:
            data.name = '这个特征没写名字'
        data = data.to_frame()
    x_name = data.name if isinstance(data, Series) else data.columns[0]
    x = alt.X(x_name, bin=alt.Bin(maxbins=40), scale=alt.Scale(padding=0), axis=alt.Axis(title=None))
    y = alt.Y('count()', axis=alt.Axis(title=None))
    tooltip = [alt.Tooltip('count()', title='count')]
    title = alt.TitleParams(text=f"{x_name}", anchor='middle')
    chart = alt.Chart(data).mark_bar(binSpacing=0).encode(x=x, y=y, tooltip=tooltip).properties(title=title,
                                                                                                width=1000,
                                                                                                height=250
                                                                                                ).interactive()
    return chart


def plot_histogram(data: Union[DataFrame, Series],
                   ) -> alt.Chart:
    """
    绘制特征的直方图。

    :param data: 用于绘图的数据
    :return:  图表对象
    """
    if isinstance(data, Series):
        return plot_single_histogram(data)
    else:
        chart_list = []
        for column in data.columns:
            chart_list.append(plot_single_histogram(data[column]))
        return alt.vconcat(*chart_list)
