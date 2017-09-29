library(tensorflow)
tf$reset_default_graph()

data(BostonHousing, package='mlbench')
y_data = BostonHousing
y_data = BostonHousing$medv; y_data = as.matrix(y_data, ncol=1)
x_data = with(BostonHousing, scale(cbind(age, crim)))

x_ = tf$placeholder(tf$float32, shape(NULL, 2L))
y_ = tf$placeholder(tf$float32, shape(NULL, 1L))
W = tf$Variable(tf$random_uniform(shape(2L, 1L), -1.0, 1.0))
b = tf$Variable(tf$zeros(shape(1L)))
y = tf$matmul(x_, W) + b

# Set up optimizer
loss = tf$reduce_mean((y - y_)^2)
optimizer = tf$train$GradientDescentOptimizer(0.1)
train = optimizer$minimize(loss)

# Start sessiom and initialize
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# minimize
for (step in 1:200) {
    sess$run(train, feed_dict=dict(x_=x_data, y_=y_data))
    print(sess$run(loss, feed_dict=dict(x_=x_data, y_=y_data)))
}

# (evaluate loss at 'optimal' W)
print(sess$run(b, feed_dict=dict(x_=x_data, y_=y_data)))
print(sess$run(W, feed_dict=dict(x_=x_data, y_=y_data)))

# do regression to check
mod = lm(medv ~ I((age-mean(age))/sd(age)) +
    I((crim-mean(crim))/sd(crim)), BostonHousing)


