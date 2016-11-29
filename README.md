# DL4J-Bugprediction - Bachelorthesis of Sébastien Broggi
Using DL4J the goal of this project is to successfully classify buggy code and suggest fixes, based on previously learned fixes in a project.
This is my Bachelor's thesis.

# Setup
To set the project up, you need Maven and a running MySQL-Server. Adapt the values in SQLConnector.java to your needs.
In my setup, I have 2 MySQL-schemes:
"bugfixes" and "bugfixes_eclipse", each containing a table "method_change" with columns "pre_context", "post_context", "old_context" and "new_context", where pre-and post-context is the unchanged part of a method and the columns new- and old-content describe the changed lines to fix the bug, before and after.

