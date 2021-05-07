<!--
 Copyright 2021 Zhongyang Zhang
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Git Frequent Operations 

1. Accidently added a big file and commited.

`git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch <File/Folder>' -- --all`

2. Remove some file just from the remote side.

`git rm -r --cached <File/Folder>`

3. Remove the file from the Git repository and the filesystem

`git rm -r <File/Folder>`

