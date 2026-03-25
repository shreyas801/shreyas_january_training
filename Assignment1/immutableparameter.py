def count_message(msg, count=0):
    count = count + 1
    print("Message:", msg)
    print("Count:", count)
    return count
c = count_message("Hello")
c = count_message("Jai sri ram", c)
c = count_message("Welcome to India", c)
