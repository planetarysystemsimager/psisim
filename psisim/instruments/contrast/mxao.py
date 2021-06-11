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
    

def load_config(config_fn):
    import configparser
    config = configparser.ConfigParser(default_section='default',
                                       inline_comment_prefixes='#',
                                       #strict=False,
                                       )
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
    test_psi()
    test_load_config()
    test_write_config()

