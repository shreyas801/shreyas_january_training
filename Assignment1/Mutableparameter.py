def add_order(order_id, orders=None):
    if orders is None:
        orders = []
    orders.append(order_id)
    return orders
history = add_order(1111)
print(history)        

history = add_order(1112, history)
print(history)       

history = add_order(1113, history)
print(history)        
