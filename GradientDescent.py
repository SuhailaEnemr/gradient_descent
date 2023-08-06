import numpy as np

# Data
X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
              55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
              45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
              48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
              78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
              55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
              60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

# spliting data :
x_train = X[: 16]
y_train = Y[: 16]
x_test = X[16:]
y_test = Y[16:]

estimated_weight, estimated_bias = gredian_descent(x_train, y_train, iterations=2000)
print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

score = MSE(y_test , x_test)
print('score : ' , score)

# Mean Squared Error
def MSE (y_true , pred) :
    cost = np.square(np.subtract(y_true,pred)).mean()
    return cost


#gradient descent
def gredian_descent (x , y , iterations = 1000 , learning_rate = 0.0001 , stop_point = 1e-4) :
    
    costs = []
    weights = []
    n = float(len(x))
    previous_cost = None
    current_weight = 0.1
    current_bias = 0.01
    
    for i in range(iterations) :
        y_pred = (current_weight * x) + current_bias
        
        current_cost = MSE(y , y_pred)
        
        if previous_cost and abs(previous_cost - current_cost) <= stop_point :
            break
            
        costs.append(current_cost)
        weights.append(current_weight)
        
        previous_cost = current_cost
        
        # calculating the gredients
        weight_derivative = -(2/n) * sum(x * (y - y_pred))
        bias_derivative = -(2/n) * sum(y - y_pred)
        
        # updating for weight and bais
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
        
    print('iteration : ' , i)
    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
    
    return current_weight, current_bias
