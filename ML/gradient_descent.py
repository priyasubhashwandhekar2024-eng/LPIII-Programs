
# Gradient Descent Algorithm

def gradient(x):
    return 2 * (x + 3)

def function(x):
    return (x + 3) ** 2

# Initial value
x = 2.0

# Learning rate
learning_rate = 0.1

# Number of iterations
iterations = 50

print("Gradient Descent Algorithm")
print("Initial x =", x)
print("\nIteration\t x\t\t y")

for i in range(iterations):
    y = function(x)
    print(f"{i}\t\t {x:.6f}\t {y:.6f}")

    # Update x using the gradient
    x = x - learning_rate * gradient(x)

print("\nLocal minimum point:")
print("x =", round(x, 6))
print("y =", round(function(x), 6))

