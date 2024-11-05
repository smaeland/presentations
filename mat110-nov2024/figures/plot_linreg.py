import numpy as np
from bokeh.layouts import column, row, layout, Spacer
from bokeh.models import ColumnDataSource, CustomJS, Slider, Span, Label, Arrow, VeeHead
from bokeh.plotting import figure, show, output_file, save

#import matplotlib.pyplot as plt

DEBUG = True

marker_size = 16
line_width = 5


def generate_data(offset=4.0, slope=2.0, num=100, seed=42):

    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(low=0, high=0.8, size=num)
    y = offset + x*slope + rng.normal(scale=0.6, size=num)

    return x, y


X_train, y_train = generate_data(num=20)


def plot_data_only():

    output_filename = 'linreg_data_only.html'
    output_file(filename=output_filename, title='Plot of data only')

    data = ColumnDataSource(data={'x': X_train, 'y': y_train})

    plot = figure(
        x_range=(0, 1.2),
        y_range=(0, 8),
        sizing_mode='stretch_both',
        tools='pan,wheel_zoom,box_zoom,reset,crosshair',
        active_drag='pan',
        active_scroll='wheel_zoom'
    )
    plot.scatter(x='x', y='y', source=data, size=marker_size)
    span = Span(location=1.0, dimension='height', line_dash="dashed")
    plot.add_layout(span)

    if DEBUG:
        show(plot)
    else:
        save(plot)


def plot_model():

    output_filename = 'linreg_sliders.html'
    output_file(filename=output_filename, title='Plot of cubic function')

    data = ColumnDataSource(data={'x': X_train, 'y': y_train})
    model = ColumnDataSource(
        data={
            'x': np.linspace(-1, 2, 10),
            'y': np.ones(10)
        }
    )
    point = ColumnDataSource(
        data={
            'px': np.array([1.0]),
            'py': np.array([-1.0])
        }
    )

    plot = figure(
        x_range=(0, 1.2),
        y_range=(0, 8),
        sizing_mode='stretch_both',
        min_width=400,
        tools='pan,wheel_zoom,box_zoom,reset,crosshair',
        active_drag='pan',
        active_scroll='wheel_zoom'
    )
    plot.scatter(x='x', y='y', source=data, size=marker_size)
    plot.line(x='x', y='y', source=model, line_width=line_width, line_alpha=0.6, line_color="#D81B60")
    plot.scatter(x='px', y='py', source=point, size=marker_size)

    span = Span(location=1.0, dimension='height', line_dash="dashed")
    plot.add_layout(span)

    w0 = Slider(start=-1, end=10, value=1, step=0.1, title="w0", bar_color="#2196F3")
    w1 = Slider(start=-10, end=10, value=0, step=0.1, title="w1", bar_color="#2196F3")

    callback = CustomJS(
        args=dict(source=model, pt=point, w0=w0, w1=w1),
        code = """
        const _w0 = w0.value
        const _w1 = w1.value

        const x = source.data.x
        const y = Array.from(x, (x) => _w0 + _w1*x)
        source.data = { x, y }

        const px = pt.data.px
        const py = Array.from(px, (px) => _w0 + _w1*px)
        pt.data = { px, py }
        """
    )

    w0.js_on_change('value', callback)
    w1.js_on_change('value', callback)

    sliders = row(Spacer(width=50), w0, w1, height=50, sizing_mode='stretch_width')
    grid = column([plot, sliders], sizing_mode='stretch_both')

    if DEBUG:
        show(grid)
    else:
        save(grid)




def plot_model_loss():

    output_filename = 'linreg_loss.html'
    output_file(filename=output_filename, title='Plot of cubic function')

    data = ColumnDataSource(data={'x': X_train, 'y': y_train})
    model = ColumnDataSource(
        data={
            'x': np.linspace(-1, 2, 10),
            'y': np.ones(10)
        }
    )

    plot = figure(
        x_range=(0, 1.2),
        y_range=(0, 8),
        sizing_mode='stretch_both',
        min_width=400,
        tools='pan,wheel_zoom,box_zoom,reset,crosshair',
        active_drag='pan',
        active_scroll='wheel_zoom'
    )
    plot.scatter(x='x', y='y', source=data, size=marker_size)
    plot.line(x='x', y='y', source=model, line_width=line_width, line_alpha=0.6, line_color="#D81B60")

    label = Label(x=0.85, y=2, text=r'$$L = $$', text_font_size='42px')
    plot.add_layout(label)

    w0 = Slider(start=-1, end=10, value=1, step=0.1, title="w0", bar_color="#2196F3")
    w1 = Slider(start=-10, end=10, value=0, step=0.1, title="w1", bar_color="#2196F3")

    callback = CustomJS(
        args=dict(source=model, data=data, label=label, w0=w0, w1=w1),
        code = """
        const _w0 = w0.value
        const _w1 = w1.value

        const x = source.data.x
        const y = Array.from(x, (x) => _w0 + _w1*x)
        source.data = { x, y }

        const x_data = data.data.x
        const y_data = data.data.y

        let sum_squared_error = 0;
        for (let i = 0; i < x_data.length; i++) {
            const error = (_w0 + _w1*x_data[i]) - y_data[i]
            sum_squared_error += error*error
        }
        const mse = sum_squared_error / x_data.length

        label.text = "$$L = $$ " + mse.toFixed(3).toString()
        """
    )

    w0.js_on_change('value', callback)
    w1.js_on_change('value', callback)

    sliders = row(Spacer(width=50), w0, w1, height=50, sizing_mode='stretch_width')
    grid = column([plot, sliders], sizing_mode='stretch_both')

    if DEBUG:
        show(grid)
    else:
        save(grid)


def plot_gradient():

    output_filename = 'linreg_gradient.html'
    output_file(filename=output_filename, title='Plot of cubic function')

    data = ColumnDataSource(data={'x': X_train, 'y': y_train})
    model = ColumnDataSource(
        data={
            'x': np.linspace(-1, 2, 10),
            'y': np.ones(10)
        }
    )

    plot1 = figure(
        x_range=(0, 1.2),
        y_range=(0, 8),
        sizing_mode='stretch_both',
        min_width=400,
        tools='pan,wheel_zoom,box_zoom,reset,crosshair',
        active_drag='pan',
        active_scroll='wheel_zoom',
        x_axis_label='x',
        y_axis_label='y'
    )
    plot1.scatter(x='x', y='y', source=data, size=marker_size)
    plot1.line(x='x', y='y', source=model, line_width=line_width, line_alpha=0.6, line_color="#D81B60")

    plot1.xaxis.axis_label_text_font_size = "32pt"
    plot1.yaxis.axis_label_text_font_size = "32pt"
    
    label = Label(x=0.85, y=2, text=r'$$L = $$', text_font_size='42px')
    plot1.add_layout(label)

    plot2 = figure(
        x_range=(-1, 10),
        y_range=(-10, 10),
        sizing_mode='stretch_both',
        min_width=400,
        x_axis_label=r'$$\theta_1$$',
        y_axis_label=r'$$\theta_2$$'
    )
    plot2.xaxis.axis_label_text_font_size = "32pt"
    plot2.yaxis.axis_label_text_font_size = "32pt"

    label1 = Label(x=6, y=-5, text=r'$$\frac{\partial L}{\partial\theta_1} = $$', text_font_size='36px')
    label2 = Label(x=6, y=-8, text=r'$$\frac{\partial L}{\partial\theta_2} = $$', text_font_size='36px')
    plot2.add_layout(label1)
    plot2.add_layout(label2)



        #xm, ym = np.meshgrid(np.linspace(-1, 10, 20), np.linspace(-10, 10, 20))
    #zm = (2/len(X_train)) * X_train.T @ (X_train @ )
    
    arrowhead = VeeHead(fill_color="#00796B")
    arrow = Arrow(end=arrowhead, x_start=0, y_start=0, x_end=1, y_end=1, line_color='#00796B', line_width=5)
    plot2.add_layout(arrow)

    w0 = Slider(start=-1, end=10, value=1, step=0.1, title="w0", bar_color="#2196F3")
    w1 = Slider(start=-10, end=10, value=0, step=0.1, title="w1", bar_color="#2196F3")

    callback = CustomJS(
        args=dict(source=model, data=data, label=label, label1=label1, label2=label2, arrow=arrow, w0=w0, w1=w1),
        code = """
        const _w0 = w0.value
        const _w1 = w1.value

        const x_data = data.data.x
        const y_data = data.data.y
        const len_data = x_data.length

        const x = source.data.x
        const y = Array.from(x, (x) => _w0 + _w1*x)
        source.data = { x, y }

        let mse = 0
        let grad_theta1 = 0
        let grad_theta2 = 0
        for (let i = 0; i < len_data; i++) {
            const error = (_w0 + _w1*x_data[i] - y_data[i])
            grad_theta1 += 2*error
            grad_theta2 += 2*error*x_data[i]
            mse += error*error
        }
        grad_theta1 /= len_data
        grad_theta2 /= len_data
        mse /= len_data

        grad_theta1 *= -1
        grad_theta2 *= -1

        arrow.x_start = _w0
        arrow.y_start = _w1
        arrow.x_end = _w0 + grad_theta1*0.75
        arrow.y_end = _w1 + grad_theta2*3

        console.log("grad_theta1 =", grad_theta1.toFixed(3), "grad_theta2 =", grad_theta2.toFixed(3))

        label.text = "$$L = $$ " + mse.toFixed(3).toString()
        label1.text = "$$\\\\frac{\\\\partial L}{\\\\partial\\\\theta_2} = $$ " + grad_theta1.toFixed(3).toString()
        label2.text = "$$\\\\frac{\\\\partial L}{\\\\partial\\\\theta_2} = $$ " + grad_theta2.toFixed(3).toString()
        """
    )

    w0.js_on_change('value', callback)
    w1.js_on_change('value', callback)

    plots = row([plot1, plot2], sizing_mode='stretch_both')
    sliders = row(Spacer(width=50), w0, w1, height=50, sizing_mode='stretch_width')
    grid = column([plots, sliders], sizing_mode='stretch_both')

    if DEBUG:
        show(grid)
    else:
        save(grid)
    




if __name__ == '__main__':

    #plot_data_only()
    #plot_model()
    #plot_model_loss()
    plot_gradient()
    
