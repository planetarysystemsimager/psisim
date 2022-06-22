# routines for using mxlib and aoSystem from Jared Males for contrast simulation
# Michael Fitzgerald (mpfitz@ucla.edu) 2021-6-10

# https://jaredmales.github.io/mxlib-doc/index.html
# https://github.com/jaredmales/mxlib
# https://github.com/jaredmales/aoSystem


import os

def run_aosystem(conf_fn,
                 aopath = os.path.expanduser('~/work/aoSystem/'),
                 aocmd = 'aoSystem',
                 ):
    import subprocess
#    with subprocess.Popen([aopath+aocmd, '-c', conf_fn],
#                          stdout=subprocess.PIPE,
#                          shell=True,
#                          text=True,
#                          ) as proc:
#        out, err = proc.communicate()
#    print('output:', out)
#    print('err:', err)
    output = subprocess.run([aopath+aocmd, '-c', conf_fn],
                            text=True,
                            capture_output=True,
                            )
    #print('output:', output.stdout)


    # FIXME  can we always turn output into a table?


    return output

def test_psi():
    conf_fn = 'PSI.conf'
    output = run_aosystem(conf_fn)
    print('output:', output.stdout)

    import astropy.table
    t = astropy.table.Table.read(output.stdout,
                                 format='ascii.commented_header',
                                 )
    print(t.colnames)
    print(t)

    import matplotlib as mpl
    import pylab
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(t['mag'], t['Strehl'])
    ax.set_xlabel('mag')
    ax.set_ylabel('Strehl')
    pylab.draw()
    pylab.show()
    

def test_modes():
    # load configuration
    config_fn = 'PSI.conf'
    config = load_config(config_fn)

    # udpate configuration
    print(config)
    #config['global']['mode'] = 'ErrorBudget'
    #config['global']['mode'] = 'C0Var'
    #config['global']['mode'] = 'C0Con'
    #config['global']['mode'] = 'CVarAll'
    #config['global']['mode'] = 'CConAll'
    #config['global']['mode'] = 'temporalPSD'
    config['global']['mode'] = 'temporalPSDGrid'

    # output configuration
    out_config_fn = 'PSI2.conf'
    write_config(out_config_fn, config)

    # run simulator
    #output = run_aosystem(config_fn) # TEMP
    output = run_aosystem(out_config_fn)
    print('output:', output.stdout)

    # parse output to table
    import astropy.table
    if config['global']['mode'] == 'ErrorBudget':
        t = astropy.table.Table.read(output.stdout,
                                     format='ascii.commented_header',
                                     )
    elif config['global']['mode'] == 'C0Var': # FIXME  handle 1,2...
        t = astropy.table.Table.read(output.stdout,
                                     format='ascii',
                                     names=['i',config['global']['mode']],
                                     )
    elif config['global']['mode'] == 'C0Con': # FIXME  handle 1,2...
        t = astropy.table.Table.read(output.stdout,
                                     format='ascii',
                                     names=['i',config['global']['mode']],
                                     )
    elif config['global']['mode'] == 'CVarAll':
        t = astropy.table.Table.read(output.stdout,
                                     format='ascii',
                                     names=['i']+['C{}Var'.format(i) for i in range(8)],
                                     )
    elif config['global']['mode'] == 'CConAll':
        t = astropy.table.Table.read(output.stdout,
                                     format='ascii',
                                     names=['i']+['C{}Con'.format(i) for i in range(8)],
                                     )
    elif config['global']['mode'] == 'temporalPSD':
        # process header output that starts with #, until we get a line with a bunch of ###
        lines = output.stdout.splitlines()
        delimited = ['#'*3 in line for line in lines] # boolean array of whether line is delimiter
        idels = [i for i, x in enumerate(delimited) if x] # list of delimiter indices
        # split header and computation output
        hdrtxt = lines[0:idels[0]]
        outtxt = lines[idels[0]+1:]
        print('\n'.join(hdrtxt))
        # parse computation output
        t = astropy.table.Table.read('\n'.join(outtxt),
                                     format='ascii.commented_header',
                                     )
    elif config['global']['mode'] == 'temporalPSDGrid':
        raise NotImplementedError
    elif config['global']['mode'] == 'temporalPSDGridAnalyze':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # show output
    print(t)


def load_config(config_fn):
    import configparser
    config = configparser.ConfigParser(default_section='default',
                                       inline_comment_prefixes='#',
                                       #strict=False,
                                       )
    config.optionxform = lambda option: option # don't convert to lowercase
    with open(config_fn) as f:
        file_content = '[global]\n' + f.read()
    config.read_string(file_content)
    return config

def test_load_config():
    config_fn = 'PSI.conf'
    config = load_config(config_fn)
    for s in config.sections():        
        print('{}: {}'.format(s, config.items(s)))


def write_config(config_fn, config):
    import io
    with io.StringIO() as file_content:
        config.write(file_content)
        content = file_content.getvalue()
    lines = content.split('\n')
    with open(config_fn, 'w') as f:
        f.write('\n'.join(lines[1:])) # strips out '[global]'

def test_write_config():
    config_fn = 'PSI.conf'
    config = load_config(config_fn)
    out_config_fn = 'PSI2.conf'
    write_config(out_config_fn, config)


if __name__=='__main__':
    #test_psi()
    #test_load_config()
    #test_write_config()
    test_modes()
