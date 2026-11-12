from . import gen_endpoints,gen_parent_gains,gen_routes
def main():
    for module in (gen_endpoints,gen_parent_gains,gen_routes):module.main();print(module.__name__,'saved',flush=True)
if __name__=='__main__':main()
