import os

log_files = []
for subdir, dirs, files in os.walk('./'):
    for file in files:
      if file.rfind('gnss_log') != -1:
          print(file)
          log_files.append(file)

for name_file in log_files:    
    lines = []
    with open(name_file, 'r') as f:
        lines = f.readlines()
    
            
    fixes = list(filter(lambda l: l.find('Fix') != -1, lines))
    header = ','.join(fixes.pop(0).split(',')[2:])
    fixes  = [','.join(fix.split(',')[2:]) for fix in fixes]
    
    with open('gps'+name_file[4:], 'w') as f:
        f.write(header)
        f.writelines(fixes)
