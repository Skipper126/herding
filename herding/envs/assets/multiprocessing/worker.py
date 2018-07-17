
class Worker:

    def __init__(self):
        self.__quit = False
        
    def quit(self):
        self.__quit = True

    def should_quit(self):
        return self.__quit
