;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(require :nik-utils)
(load "xml-stuff.lisp")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;    ___                          __ _ _                                      
;;   / __\ __ _ _ __   ___ __ _   / _(_) | ___ _ __   __ _ _ __ ___   ___  ___ 
;;  /__\/// _` | '_ \ / __/ _` | | |_| | |/ _ \ '_ \ / _` | '_ ` _ \ / _ \/ __|
;; / \/  \ (_| | | | | (_| (_| | |  _| | |  __/ | | | (_| | | | | | |  __/\__ \
;; \_____/\__,_|_| |_|\___\__,_| |_| |_|_|\___|_| |_|\__,_|_| |_| |_|\___||___/
;;                                                                             
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter important-indices (list 0 1 2 3 6)) ;; Equivalent to real-id, gender, group, session, photo count

(defun banca-important(lst)
  "Only keep the parts that are important for us"
  (loop for important-index in important-indices
     collect (nth important-index lst)))

(defun split-banca(line)
  "Split a filename (line) into its important parts"
  (banca-important (cl-ppcre::split "(_|\\.)" (nik-utils::trim-all-white line))))

(defun banca-part(line index)
  "Pick out only the index (column) of a filename (line)"
  (nth index (split-banca line)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;     ___      _              _                   _                       
;;    /   \__ _| |_ __ _   ___| |_ _ __ _   _  ___| |_ _   _ _ __ ___  ___ 
;;   / /\ / _` | __/ _` | / __| __| '__| | | |/ __| __| | | | '__/ _ \/ __|
;;  / /_// (_| | || (_| | \__ \ |_| |  | |_| | (__| |_| |_| | | |  __/\__ \
;; /___,' \__,_|\__\__,_| |___/\__|_|   \__,_|\___|\__|\__,_|_|  \___||___/
;;                                                                         
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Datastruct to quickly handle unique numbers for filename
(let ((my-table (make-hash-table :test 'equal)) ;; strings require special test
      (file-cnt 0))

  (defun insert-filename(filename)
    (setf (gethash filename my-table) (incf file-cnt)))

  (defun filename->index(filename)
    (gethash filename my-table))

  (defun delete-filename(filename)
    (remhash filename my-table)
    (decf file-cnt))

  (defmacro with-filenames((filename) &body body)
    `(loop for ,filename being the hash-keys of ,my-table
	do (progn ,@body))))

(defun clear()
  "WARNING, we cannot just throw away the old one because the macro above will not work with function below"
  (with-filenames (filename)
    (delete-filename filename)))

;; populate with filenames
(defun populate-map(lst-of-filenames)
  (clear)
  (loop for filename in lst-of-filenames
     do (insert-filename filename)))

;; ------------------------------------------------------------------------------------------------
;; keep the splits-unique map -- Important as hell but hard to explain, example soon TODO NIK !!
(defparameter *splits-unique* nil) ;; all the unique parts of the split

;; -----------------------------
(defun visualize()
  "Visualize the unique columns"
  (labels ((max-length (lst-lst)
	     (apply #'max (mapcar #'length lst-lst)))

	   (nil-safe (element)
	     (if (eq nil element)
		 ""
		 element)))

    (loop for index from 0 to (1- (max-length *splits-unique*))
       do (format t "~{~5a ~}~%" (mapcar #'nil-safe (mapcar #'(lambda(lst)(nth index lst)) *splits-unique*))))))

;; -----------------------------
(defun unique-columns(lst-lst)
  "For each list in the list, keep only the unique elements (strings)"
  (loop for dim from 0 to (1- (length (first lst-lst)))
     collect (remove-duplicates (mapcar #'(lambda(lst)(nth dim lst)) lst-lst) 
				:test #'string-equal)))

;; -----------------------------
(defun filename->part-indices(filename &key (offset 0))
  "The offset is to add +1 if we are pointing into an array that starts from 1 instead of 0"
  (let ((parts (split-banca filename)))
    (loop 
       for part in parts
       for index from 0
       collect (+ offset (nik-utils::index-of part (nth index *splits-unique*))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;    __      _       _   _                 _     _           
;;   /__\ ___| | __ _| |_(_) ___  _ __  ___| |__ (_)_ __  ___ 
;;  / \/// _ \ |/ _` | __| |/ _ \| '_ \/ __| '_ \| | '_ \/ __|
;; / _  \  __/ | (_| | |_| | (_) | | | \__ \ | | | | |_) \__ \
;; \/ \_/\___|_|\__,_|\__|_|\___/|_| |_|___/_| |_|_| .__/|___/
;;                                                 |_|        
;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun column->relation(column &key (only-filenames nil))
  "To reverse the relationship normal relationship of filename -> image. We want column value -> filename*"
  (let* ((targets    (nth column *splits-unique*))
	 (datastruct (nik-utils::list-of-nils (length targets))))

    ;; populate the datastruct (mapping filenames)
    ;; we give to options, eithere we store the real filename or we save the index of the filename
    (with-filenames (filename)
      (if only-filenames
	  (push filename                   (nth (nik-utils::index-of (banca-part filename  column) targets) datastruct))
	  (push (filename->index filename) (nth (nik-utils::index-of (banca-part filename  column) targets) datastruct))))

    ;; return the datastruct
    (list targets datastruct)))

(defun all-relationships->strings()
  "WARNING, we are adding ones"
  (let ((lst nil))
    (with-filenames (filename)
      (push (format nil "~{~a ~}" (filename->part-indices filename :offset 1)) lst))
    lst))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;    __                            _                
;;   /__\   _  ___    ___ ___ _ __ | |_ ___ _ __ ___ 
;;  /_\| | | |/ _ \  / __/ _ \ '_ \| __/ _ \ '__/ __|
;; //__| |_| |  __/ | (_|  __/ | | | ||  __/ |  \__ \
;; \__/ \__, |\___|  \___\___|_| |_|\__\___|_|  |___/
;;      |___/                                        
;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; example usage:
;; (get-eyecenters (nik-utils::search-files-in-folder "/mnt/jupiter/databases_raw/BANCA_PGM_IMAGES/eyecenter/" "*.pos"))
;; (nik-utils::random-elements 10 (nik-utils::search-files-in-folder "/mnt/jupiter/databases_raw/BANCA_PGM_IMAGES/eyecenter/" "*.pos"))

(defun filename->eyecenter-filename(filename)
  (concatenate 'string "/mnt/jupiter/databases_raw/BANCA_PGM_IMAGES/eyecenter/" (pathname-name filename) ".pos"))

(defun xyxy->hwhw(lst)
  "In torch we do not work with x,y coords but h,w coords"
  (list (nth 1 lst) (nth 0 lst) (nth 3 lst) (nth 2 lst)))

(defun file->eyecenter(filename)
  (nik-utils::list-to-string 
   (xyxy->hwhw 
    (nik-utils::split-line 
     (nik-utils::file->string filename)))))

#|
(defun get-eyecenters(pathnames)
  (let ((lst nil))
    (loop for file in pathnames
       do (push (list (pathname-name file) (file->eyecenter file)) lst))

    ;; return the list
    (reverse lst)))
|#

(defun get-all-eyecenters()
  (let ((lst nil))
    (with-filenames (filename)
      (push (file->eyecenter (filename->eyecenter-filename filename)) lst))

    ;; return the list
    (reverse lst)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               _       
;;   /\/\   __ _(_)_ __  
;;  /    \ / _` | | '_ \ 
;; / /\/\ \ (_| | | | | |
;; \/    \/\__,_|_|_| |_|
;;                       
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun main(filename save-filename &key (image-path "/mnt/jupiter/databases_raw/BANCA_PGM_IMAGES/") (eyecenters t)) 
  (let ((filenames (nik-utils::file->strings filename))
	(splits-tmp))

    ;; create the individual splits
    (setf splits-tmp      (mapcar #'split-banca filenames))
    (setf *splits-unique* (unique-columns splits-tmp))
    
    ;; populate the map
    (populate-map filenames)

    ;; create the dataset basics (filenames and the userids)
    (xml-to-file (make-dataset  (make-instance 'Location
					       :pathnames (list image-path))

				(make-instance 'ExternalArraySet 
					       :etype "uint8"
					       :shape "576 720"
					       :content filenames))

				#|
				(make-instance 'ArraySet
						 :role "EyeCenters"
						 :etype "uint32"
						 :shape "4"
						 :content (get-all-eyecenters)))
				|#
 		 save-filename)))

